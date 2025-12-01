#!/usr/bin/env python3
"""从 Excel 第三列读取网址，抓取页面并在段落中检测关键词，输出到新的 Excel 文件。"""
import argparse
import concurrent.futures
import re
import time
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from urllib.parse import urlparse
from pathlib import Path
import xlsxwriter


def normalize_url(url: str) -> str:
    url = str(url).strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "http://" + url
    return url


def read_urls_from_excel(path: str, sheet_name: str | None, url_col: int) -> List[tuple]:
    """按原表顺序返回每行的三元组 (row_index, normalized_url_or_empty, original_first_column_value)。
    保留所有行（包括 URL 为空的行），url_col 为 1-based。
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    # 如果用户传入 sheet_name=None，pandas 会返回一个 dict；选取第一个 sheet
    if isinstance(df, dict):
        df = next(iter(df.values()))
    # 重置索引以使用连续的行号，便于按顺序输出
    df = df.reset_index(drop=True)
    col_idx = url_col - 1
    if col_idx < 0:
        raise ValueError(f"指定的列超出范围：{url_col}")

    results: List[tuple] = []
    ncols = df.shape[1]
    for i in range(df.shape[0]):
        url_val = ""
        if col_idx < ncols:
            v = df.iat[i, col_idx]
            if not pd.isna(v):
                url_val = str(v).strip()
        norm = normalize_url(url_val) if url_val else ""
        first_col_val = ""
        second_col_val = ""
        if ncols >= 1:
            f0 = df.iat[i, 0]
            if not pd.isna(f0):
                first_col_val = str(f0)
        if ncols >= 2:
            f1 = df.iat[i, 1]
            if not pd.isna(f1):
                second_col_val = str(f1)
        results.append((i, norm, first_col_val, second_col_val))
    return results


def fetch_page(session: requests.Session, url: str, timeout: float = 10.0) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KeywordCrawler/1.0; +https://example.com)"
    }
    tries = 3
    for i in range(tries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return resp.text
        except RequestException:
            if i < tries - 1:
                time.sleep(1 + i)
                continue
            return None


def find_matching_paragraphs(html: str, keywords_re: re.Pattern) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    matches: List[str] = []
    # 首先在 <p> 标签中查找
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text and keywords_re.search(text):
            matches.append(text)
    # 如果没有 p 标签匹配，可以考虑在正文文本中按段落分割搜索
    if not matches:
        body = soup.body
        if body:
            body_text = body.get_text("\n", strip=True)
            for para in [x.strip() for x in body_text.split("\n") if x.strip()]:
                if keywords_re.search(para):
                    matches.append(para)
    return matches


def process_single(idx: int, url: str, keywords_re: re.Pattern, session: requests.Session, timeout: float) -> tuple:
    """处理单个 URL，返回 (idx, matches_list)。如果 url 为空或请求失败，matches_list 为空列表。"""
    if not url:
        return idx, []
    html = fetch_page(session, url, timeout=timeout)
    if not html:
        return idx, []
    matches = find_matching_paragraphs(html, keywords_re)
    return idx, matches


def build_keywords_pattern(keywords: List[str]) -> re.Pattern:
    # 转义并按 | 连接，忽略大小写
    escaped = [re.escape(k.strip()) for k in keywords if k and k.strip()]
    if not escaped:
        # 不会匹配任何内容
        return re.compile(r"(?!x)x")
    pattern = r"(" + r"|".join(escaped) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description="从 Excel 读取 URL，检测页面段落中的关键词，输出到新的 Excel 文件。")
    script_dir = Path(__file__).parent
    default_input = script_dir / "input.xlsx"
    default_output = script_dir / "result.xlsx"
    parser.add_argument("--input", "-i", default=str(default_input), help="输入 Excel 文件路径（默认同级目录下的 input.xlsx）")
    parser.add_argument("--sheet", "-s", default=None, help="工作表名称或索引（可选）")
    parser.add_argument("--url-col", "-c", type=int, default=3, help="包含 URL 的列（1-based，默认 3）")
    parser.add_argument("--keywords", "-k", default="留学生,外国,来华,赴华", help="逗号分隔的关键词列表，例如：留学生,签证")
    parser.add_argument("--keywords-file", "-K", default=None, help="每行一个关键词的文件（优先于 --keywords）")
    parser.add_argument("--output", "-o", default=str(default_output), help="输出 Excel 文件路径（默认同级目录下的 result.xlsx）")
    parser.add_argument("--workers", "-w", type=int, default=5, help="并发线程数，默认 5")
    parser.add_argument("--timeout", type=float, default=10.0, help="请求超时时间（秒），默认 10")
    args = parser.parse_args()

    if args.keywords_file:
        with open(args.keywords_file, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
    elif args.keywords:
        keywords = [k.strip() for k in re.split(r"[,;]", args.keywords) if k.strip()]
    else:
        print("请通过 --keywords 或 --keywords-file 提供关键词。")
        return

    try:
        url_rows = read_urls_from_excel(args.input, args.sheet, args.url_col)
    except Exception as e:
        print(f"读取 Excel 出错: {e}")
        return

    if not url_rows:
        print("未从 Excel 中读取到任何 URL。请检查输入文件和列索引。")
        return

    keywords_re = build_keywords_pattern(keywords)

    session = requests.Session()
    total_rows = len(url_rows)
    print(f"开始处理 {total_rows} 行，使用 {args.workers} 个线程...")

    # 准备结果映射，初始化为空列表（表示未检出）
    results_map: dict[int, List[str]] = {i: [] for i, _, _, _ in url_rows}

    # 提交有 URL 的任务
    to_submit = []
    for i, url, _, _ in url_rows:
        if url:
            to_submit.append((i, url))

    processed_count = total_rows - len(to_submit)  # 已处理（没有 URL 的行）
    if processed_count > 0:
        print(f"{processed_count} 行没有 URL，已标记为未检出。")

    start_time = time.time()
    progress_step = max(1, total_rows // 20)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(process_single, i, url, keywords_re, session, args.timeout): i for i, url in to_submit}
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                _idx, matches = fut.result()
                results_map[_idx] = matches or []
            except Exception as e:
                print(f"任务异常（行 {idx}）: {e}")
                results_map[idx] = []
            processed_count += 1
            if processed_count % progress_step == 0 or processed_count == total_rows:
                elapsed = time.time() - start_time
                print(f"已处理 {processed_count}/{total_rows} 行，耗时 {elapsed:.1f}s")

    # 根据原表顺序构建输出行，保持行数和顺序不变
    out_rows: List[tuple] = []
    for i, url, first_col, second_col in url_rows:
        matches = results_map.get(i, [])
        if matches:
            # 编号并合并为单元格内容，段落间用两个换行分隔
            numbered = "\n\n".join(f"{j+1}. {m}" for j, m in enumerate(matches))
            out_rows.append((first_col, url, numbered, second_col))
        else:
            out_rows.append((first_col, url, "未检出", second_col))

    total_elapsed = time.time() - start_time
    print(f"处理完成，共处理 {total_rows} 行，耗时 {total_elapsed:.1f}s")

    # 使用 xlsxwriter 写入并对关键词做标红（单元格内局部红色）
    try:
        workbook = xlsxwriter.Workbook(args.output)
        worksheet = workbook.add_worksheet()
        # 单元格格式：自动换行、水平居中、垂直居中
        wrap_format = workbook.add_format({'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
        # 关键词使用红色字体（不改变对齐，由 wrap_format 提供）
        red_fmt = workbook.add_format({'font_color': 'red'})

        # 可选：设置列宽以便显示
        worksheet.set_column(0, 0, 20)  # original first column
        worksheet.set_column(1, 1, 40)  # url
        worksheet.set_column(2, 2, 80)  # matched_paragraph
        worksheet.set_column(3, 3, 20)  # original second column

        # helper: 将一个段落按关键词分割为带格式的序列
        def paragraph_to_rich_args(paragraph: str):
            parts = []
            last = 0
            for m in keywords_re.finditer(paragraph):
                s, e = m.span()
                if s > last:
                    parts.append(paragraph[last:s])
                parts.append((paragraph[s:e], True))
                last = e
            if last < len(paragraph):
                parts.append(paragraph[last:])
            # 转换为 xlsxwriter write_rich_string 所需的参数序列
            args = []
            for seg in parts:
                if isinstance(seg, tuple) and seg[1] is True:
                    # keyword segment: format then string
                    args.append(red_fmt)
                    args.append(seg[0])
                else:
                    args.append(seg)
            return args

        # 写入每一行（无表头，保持与原表相同行数和顺序）
        for row_idx, (i, url, first_col, second_col) in enumerate(url_rows):
            worksheet.write(row_idx, 0, first_col, wrap_format)
            worksheet.write(row_idx, 1, url, wrap_format)
            matches = results_map.get(i, [])
            if not matches:
                worksheet.write(row_idx, 2, "未检出", wrap_format)
            else:
                # 构建 rich string：为每段编号并对关键词标红，段落之间用两个换行
                rich_args = []
                for p_idx, para in enumerate(matches):
                    prefix = f"{p_idx+1}. "
                    rich_args.append(prefix)
                    seg_args = paragraph_to_rich_args(para)
                    # 如果段落以关键字开始，seg_args 第一个元素会是 format；为保证 write_rich_string 参数合法，
                    # 我们需要确保 rich_args 最初是字符串（可以是空字符串）
                    rich_args.extend(seg_args)
                    if p_idx != len(matches) - 1:
                        rich_args.append("\n\n")

                # write_rich_string 要求至少一个字符串参数；若首项为 format，写入时会报错，因此确保首项为字符串
                if isinstance(rich_args[0], (str,)):
                    worksheet.write_rich_string(row_idx, 2, *rich_args, wrap_format)
                else:
                    # 在最前面加空字符串
                    worksheet.write_rich_string(row_idx, 2, "", *rich_args, wrap_format)

            worksheet.write(row_idx, 3, second_col, wrap_format)

        workbook.close()
        print(f"输出已写入: {args.output}")
    except Exception as e:
        print(f"写入 Excel 失败: {e}")


if __name__ == "__main__":
    main()

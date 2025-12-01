# 爬虫：从 Excel 读取 URL 并检测包含关键词的段落

此脚本从指定 Excel 文件的某一列读取网址，抓取每个页面并查找包含指定关键词的段落，最后把结果输出为新的 Excel 文件（无表头）。

关键特性（已实现）
- 默认直接运行 `python crawl_keywords.py` 等同于：
	`python crawl_keywords.py -i input.xlsx -c 3 -k "留学生,外国" -o result.xlsx`
	其中 `input.xlsx` 和 `result.xlsx` 位于脚本同级目录。
- 输出保持与原表相同的行数与顺序（包括 URL 为空的行）。
- 输出为无表头的 Excel，列顺序如下：
	1) 原表第一列
	2) URL（来自指定的列）
	3) 匹配结果（单元格内按序号合并段落；若无匹配则写入 `未检出`）
	4) 原表第二列
- 对单个链接检测到多个段落时，段落在同一单元格内按序号合并，段落之间用两个换行分隔；匹配到的关键词在单元格内以红色字体高亮（局部标红）。
- 所有输出单元格默认设置为：自动换行、水平居中、垂直居中。

依赖
- Python 3.8+
- 安装项目依赖（推荐）：
```powershell
python -m pip install -r requirements.txt
```
requirements.txt 中包含：`pandas`, `openpyxl`, `requests`, `beautifulsoup4`, `lxml`, `XlsxWriter`。

运行示例（PowerShell）
- 使用默认同级文件（在 `d:\Code\Python\留学生信息检测` 中）：
```powershell
cd d:\Code\Python\留学生信息检测
python crawl_keywords.py
```

- 指定文件与关键词：
```powershell
python d:\Code\Python\留学生信息检测\crawl_keywords.py -i d:\path\to\your_input.xlsx -c 3 -k "留学生,签证" -o d:\path\to\out.xlsx
```

参数说明
- `--input` / `-i`：输入 Excel 路径（默认脚本同级目录的 `input.xlsx`）。
- `--sheet` / `-s`：工作表名称或索引（可选）。
- `--url-col` / `-c`：包含 URL 的列，1-based（默认 3）。
- `--keywords` / `-k`：逗号分隔的关键词列表（默认 `留学生,外国`）。
- `--keywords-file` / `-K`：每行一个关键词的文件，优先于 `--keywords`。
- `--output` / `-o`：输出文件路径（默认脚本同级目录的 `result.xlsx`）。
- `--workers` / `-w`：并发线程数（默认 5）。
- `--timeout`：请求超时（秒，默认 10）。


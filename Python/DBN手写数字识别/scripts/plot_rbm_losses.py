"""
解析训练控制台输出，提取 RBM0/RBM1/RBM2 每轮 loss 并生成 PNG 图与 CSV。

用法（在项目根目录运行）：
    python scripts/plot_rbm_losses.py \
        --input "(vscode) DCodeGithubOtherPythonDBN手.txt" \
        --outdir model

默认会将结果保存到 `model/rbm_pretrain_loss_<timestamp>.png` 和 `.csv`。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt


RBM_LINE_RE = re.compile(r"RBM\s*(\d+)\s*epoch\s*(\d+)/\d+\s*loss=([0-9]*\.?[0-9]+)")


def try_register_bundled_font():
    """若仓库内存在 `src/utils/msyh.ttc`，尝试注册供 matplotlib 使用以显示中文。"""
    bundle_path = os.path.join("src", "utils", "msyh.ttc")
    if os.path.exists(bundle_path):
        try:
            matplotlib.font_manager.fontManager.addfont(bundle_path)
            # 设置首选 sans-serif 为 Microsoft YaHei 名称（Windows 系统通常识别）
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
            print(f"已注册仓库捆绑字体用于中文显示: {bundle_path}")
        except Exception as e:
            print(f"警告：注册捆绑字体失败：{e}")


def parse_log(input_path: str):
    """解析日志文件，返回 {rbm_index: {epoch: loss}} 结构。"""
    data = defaultdict(dict)
    max_epoch = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RBM_LINE_RE.search(line)
            if m:
                rbm = int(m.group(1))
                epoch = int(m.group(2))
                loss = float(m.group(3))
                data[rbm][epoch] = loss
                if epoch > max_epoch:
                    max_epoch = epoch
    return data, max_epoch


def save_csv(out_csv: str, data: dict[int, dict[int, float]], max_epoch: int):
    # header: epoch, rbm0, rbm1, rbm2, ...
    rbm_indices = sorted(data.keys())
    header = ["epoch"] + [f"rbm{idx}" for idx in rbm_indices]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for e in range(1, max_epoch + 1):
            row = [e]
            for idx in rbm_indices:
                val = data.get(idx, {}).get(e, "")
                row.append(val)
            writer.writerow(row)


def plot_losses(out_png: str, data: dict[int, dict[int, float]], max_epoch: int):
    plt.figure(figsize=(9, 6))
    rbm_indices = sorted(data.keys())
    for idx in rbm_indices:
        epochs = sorted(data[idx].keys())
        losses = [data[idx][ep] for ep in epochs]
        # 不显示每个点的标记，只画平滑曲线；图例使用中文标签
        plt.plot(epochs, losses, label=f"RBM层 {idx}")

    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.title("RBM 预训练损失曲线")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="从训练控制台输出解析 RBM 损失并绘图")
    parser.add_argument("--input", "-i", default="(vscode) DCodeGithubOtherPythonDBN手.txt",
                        help="训练控制台输出文本文件路径（相对于仓库根或绝对路径）")
    parser.add_argument("--outdir", "-o", default="model", help="保存 PNG/CSV 的输出目录")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        # 相对到当前工作目录（仓库根）
        input_path = os.path.join(os.getcwd(), input_path)

    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        sys.exit(2)

    try_register_bundled_font()

    data, max_epoch = parse_log(input_path)
    if not data:
        print("未在日志中发现 RBM 损失行。请确认日志格式类似 'RBM 0 epoch 1/50 loss=0.123'。")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(args.outdir, f"rbm_pretrain_loss_{ts}.png")
    out_csv = os.path.join(args.outdir, f"rbm_pretrain_loss_{ts}.csv")

    save_csv(out_csv, data, max_epoch)
    plot_losses(out_png, data, max_epoch)

    print("生成完成:")
    print("  CSV:", out_csv)
    print("  PNG:", out_png)


if __name__ == "__main__":
    main()

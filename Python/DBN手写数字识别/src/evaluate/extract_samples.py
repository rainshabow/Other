"""从 MNIST 训练集中为每个类别抽取样本并保存到工程目录。

用法（在工程根目录运行）：
    python src\extract_samples.py

该脚本会在 `sample_test/<digit>/` 下保存 PNG 图片并生成 labels CSV。
"""
import os
import csv
import argparse
from PIL import Image
import torch
from torchvision import datasets


def main(out_dir='sample_test', per_class=10, data_root='data'):
    os.makedirs(out_dir, exist_ok=True)
    print(f'正在下载/加载 MNIST 训练集以生成样本（root={data_root}）...')
    train = datasets.MNIST(root=data_root, train=True, download=True)
    data = train.data  # tensor [N, H, W]
    targets = train.targets  # tensor [N]

    labels_csv_path = os.path.join(out_dir, 'labels.csv')
    rows = []

    for digit in range(10):
        idxs = (targets == digit).nonzero(as_tuple=False).view(-1).tolist()
        if len(idxs) < per_class:
            raise RuntimeError(f'样本不足：数字 {digit} 仅找到 {len(idxs)} 个样本，无法抽取 {per_class} 个')
        digit_dir = os.path.join(out_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        for i, idx in enumerate(idxs[:per_class]):
            img_tensor = data[idx]  # H x W uint8
            pil = Image.fromarray(img_tensor.numpy(), mode='L')
            filename = f'{digit}_{i}.png'
            path = os.path.join(digit_dir, filename)
            pil.save(path)
            relpath = os.path.join(str(digit), filename)
            rows.append([relpath, str(digit)])

    # write labels csv
    with open(labels_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(rows)

    print(f'完成：已在 {out_dir}/ 下保存 {per_class*10} 张样本')
    print(f'标签 CSV：{labels_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='sample_test', help='保存输出的目录（相对于工程根）')
    parser.add_argument('--per_class', type=int, default=10, help='每类要抽取的样本数量')
    parser.add_argument('--data_root', default='data', help='MNIST 数据目录（若不存在会下载到该目录）')
    args = parser.parse_args()
    main(out_dir=args.out, per_class=args.per_class, data_root=args.data_root)

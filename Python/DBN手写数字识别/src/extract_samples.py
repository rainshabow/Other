"""Extract 10 samples per class from MNIST training set and save to project directory.

Usage (from project root):
    python src\extract_samples.py

This will create `sample_test/<digit>/` directories and save PNGs and a labels CSV.
"""
import os
import csv
import argparse
from PIL import Image
import torch
from torchvision import datasets


def main(out_dir='sample_test', per_class=10, data_root='data'):
    os.makedirs(out_dir, exist_ok=True)
    print(f'Downloading/Loading MNIST training set to make samples (root={data_root})...')
    train = datasets.MNIST(root=data_root, train=True, download=True)
    data = train.data  # tensor [N, H, W]
    targets = train.targets  # tensor [N]

    labels_csv_path = os.path.join(out_dir, 'labels.csv')
    rows = []

    for digit in range(10):
        idxs = (targets == digit).nonzero(as_tuple=False).view(-1).tolist()
        if len(idxs) < per_class:
            raise RuntimeError(f'Not enough samples for digit {digit} (found {len(idxs)})')
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

    print(f'Success: saved {per_class*10} samples under {out_dir}/')
    print(f'Labels CSV: {labels_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='sample_test', help='output directory under project root')
    parser.add_argument('--per_class', type=int, default=10, help='number of samples per class')
    parser.add_argument('--data_root', default='data', help='MNIST data root (will be downloaded there)')
    args = parser.parse_args()
    main(out_dir=args.out, per_class=args.per_class, data_root=args.data_root)

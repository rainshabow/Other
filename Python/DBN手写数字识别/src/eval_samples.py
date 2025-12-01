"""Evaluate saved model on images in sample_test/ produced by extract_samples.py

Usage:
    python src\eval_samples.py --model models/dbn_mnist.pth --samples sample_test

This script will load each image under samples/<digit>/, run model prediction (no invert),
and produce a CSV with results and a simple accuracy report.
"""
import os
import argparse
import csv
import torch
import numpy as np
from PIL import Image

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    layer_sizes = ck.get('layer_sizes', None)
    return ck, layer_sizes

def build_model_from_ck(ck, layer_sizes, device):
    from dbn.model import DBN
    if layer_sizes is None:
        layer_sizes = [784, 1000, 500, 250]
    model = DBN(layer_sizes)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_pil(img, invert=False):
    # img: PIL grayscale
    img = img.resize((28,28))
    if invert:
        from PIL import ImageOps
        img = ImageOps.invert(img)
    arr = np.array(img).astype(np.float32)/255.0
    arr = arr.reshape(1, -1)
    return torch.from_numpy(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--samples', default='sample_test')
    parser.add_argument('--output', default='models/sample_eval.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck, layer_sizes = load_checkpoint(args.model, device)
    model = build_model_from_ck(ck, layer_sizes, device)

    rows = []
    total = 0
    correct = 0
    per_class = {str(i): {'tp':0, 'n':0} for i in range(10)}

    for label in sorted(os.listdir(args.samples)):
        class_dir = os.path.join(args.samples, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(class_dir, fname)
            try:
                img = Image.open(path).convert('L')
            except Exception as e:
                print('Failed to open', path, e)
                continue
            # sample_test images were saved from MNIST raw arrays; do NOT invert
            tensor = preprocess_pil(img, invert=False).to(device)
            with torch.no_grad():
                out = model(tensor)
                pred = int(out.argmax(dim=1).item())
            rows.append([path, label, pred])
            total += 1
            per_class[label]['n'] += 1
            if pred == int(label):
                correct += 1
                per_class[label]['tp'] += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path','true','pred'])
        writer.writerows(rows)

    acc = 100.0 * correct / total if total>0 else 0.0
    print(f'Sample test accuracy: {acc:.2f}% ({correct}/{total})')
    print('Per-class:')
    for k in sorted(per_class.keys(), key=lambda x:int(x)):
        stats = per_class[k]
        n = stats['n']
        tp = stats['tp']
        acc_k = 100.0*tp/n if n>0 else 0.0
        print(f' {k}: {acc_k:.2f}% ({tp}/{n})')
    print('Predictions saved to', args.output)

if __name__ == '__main__':
    main()

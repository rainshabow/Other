"""Evaluate a saved DBN model on the MNIST test set and plot results.

Saves:
- models/eval_confusion_<timestamp>.png  (confusion matrix heatmap)
- models/eval_perclass_<timestamp>.png  (per-class accuracy bar chart)
- models/eval_predictions_<timestamp>.csv (optional predictions CSV)

Usage:
    python src\evaluate.py --model models/dbn_mnist.pth
"""
import os
import argparse
import datetime
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_model_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    layer_sizes = ck.get('layer_sizes', None)
    return ck, layer_sizes

def build_model_from_ck(ck, layer_sizes, device):
    # import locally to avoid circular issues
    from dbn.model import DBN
    if layer_sizes is None:
        # fallback default
        layer_sizes = [784, 1000, 500, 250]
    model = DBN(layer_sizes)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate(model, device, batch_size=256, output_dir='models', save_csv=True):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    num_classes = 10
    conf = np.zeros((num_classes, num_classes), dtype=int)
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            t = y.cpu().numpy()
            for tt, pp in zip(t, p):
                conf[int(tt), int(pp)] += 1
            preds_all.extend(p.tolist())
            targets_all.extend(t.tolist())

    total = conf.sum()
    correct = np.trace(conf)
    acc = 100.0 * correct / total if total > 0 else 0.0
    per_class = [(100.0 * conf[i, i] / conf[i].sum()) if conf[i].sum() > 0 else 0.0 for i in range(num_classes)]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # plot confusion matrix
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
            xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)),
            ylabel='True label', xlabel='Predicted label', title=f'Confusion matrix (acc={acc:.2f}%)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # annotate
    thresh = conf.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax1.text(j, i, format(conf[i, j], 'd'), ha='center', va='center',
                     color='white' if conf[i, j] > thresh else 'black')
    conf_path = os.path.join(output_dir, f'eval_confusion_{timestamp}.png')
    fig1.tight_layout()
    fig1.savefig(conf_path)

    # plot per-class accuracy
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(range(num_classes), per_class)
    ax2.set_xticks(range(num_classes))
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-class accuracy')
    perclass_path = os.path.join(output_dir, f'eval_perclass_{timestamp}.png')
    fig2.tight_layout()
    fig2.savefig(perclass_path)

    csv_path = None
    if save_csv:
        csv_path = os.path.join(output_dir, f'eval_predictions_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'true', 'pred'])
            for i, (t, p) in enumerate(zip(targets_all, preds_all)):
                writer.writerow([i, int(t), int(p)])

    print(f'Evaluation done. Overall accuracy: {acc:.2f}%')
    print(f'Confusion matrix saved to: {conf_path}')
    print(f'Per-class accuracy saved to: {perclass_path}')
    if csv_path:
        print(f'Predictions CSV saved to: {csv_path}')

    return {'accuracy': acc, 'confusion_path': conf_path, 'perclass_path': perclass_path, 'csv_path': csv_path}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to saved model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--no_csv', action='store_true', help='do not save predictions CSV')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck, layer_sizes = load_model_checkpoint(args.model, device)
    model = build_model_from_ck(ck, layer_sizes, device)
    evaluate(model, device, batch_size=args.batch_size, output_dir=args.output_dir, save_csv=not args.no_csv)

if __name__ == '__main__':
    main()

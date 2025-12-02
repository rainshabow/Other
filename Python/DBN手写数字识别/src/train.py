import argparse
import os
import sys
import torch
# Ensure local 'src' is on sys.path so `from dbn.model import DBN` works when
# running `python src\train.py` from project root.
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = ROOT
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dbn.model import DBN
import matplotlib.pyplot as plt
import datetime

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Stage: Loading MNIST dataset...')
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # save outputs (models/plots/csv) to a `model` folder next to this train.py file
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    layer_sizes = getattr(args, 'layers', [784, 500, 200, 50])
    print(f'Stage: Initializing DBN with layer sizes: {layer_sizes}')
    dbn = DBN(layer_sizes)
    dbn.to(device)

    # pretrain
    print('Stage: Pretraining RBMs...')
    dbn.pretrain(train_loader, epochs=args.pretrain_epochs, lr=args.pretrain_lr, device=device)

    # transfer pretrained rbm weights into classifier to initialize fine-tuning
    try:
        print('Stage: Transferring RBM weights to classifier initialization...')
        dbn.transfer_rbm_to_classifier()
    except Exception as e:
        print('Warning: transfer of RBM weights failed:', e)

    # fine-tune classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dbn.classifier.parameters(), lr=args.lr)
    print('Stage: Fine-tuning classifier...')
    accuracies = []
    # track per-class accuracy history (10 classes)
    per_class_history = [[] for _ in range(10)]
    for epoch in range(args.epochs):
        dbn.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = dbn(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"Stage: Fine-tune epoch {epoch+1}/{args.epochs} loss={total_loss/n:.6f}")
        acc, per_class = test(dbn, test_loader, device)
        accuracies.append(acc)
        # append per-class accuracies (percent) to history
        for i in range(len(per_class_history)):
            per_class_history[i].append(per_class[i] if i < len(per_class) else 0.0)

    os.makedirs(base_dir, exist_ok=True)
    try:
        model_path = os.path.join(base_dir, 'dbn_mnist.pth')
        dbn.save(model_path)
        abs_model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            print(f'Model saved to {abs_model_path}')
        else:
            print(f'Warning: attempted to save model but file not found at {abs_model_path}')
    except Exception as e:
        print('Failed to save model:', e)

    # 绘制并保存准确率-轮次图
    try:
        os.makedirs(base_dir, exist_ok=True)
        epochs = list(range(1, len(accuracies)+1))
        plt.figure()
        # plot without markers to avoid clutter when many epochs
        plt.plot(epochs, accuracies, linewidth=1.5)
        plt.title('Test Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(base_dir, f'accuracy_{timestamp}.png')
        plt.savefig(out_path)
        plt.close()
        abs_out_path = os.path.abspath(out_path)
        if os.path.exists(out_path):
            print(f'Accuracy plot saved to {abs_out_path}')
        else:
            print(f'Warning: accuracy plot not found after save at {abs_out_path}')
    except Exception as e:
        print('Failed to save accuracy plot:', e)

    # 绘制并保存每类准确率变化曲线（不同颜色、无标点）
    try:
        epochs = list(range(1, len(accuracies)+1))
        plt.figure(figsize=(8, 5))
        cmap = plt.get_cmap('tab10')
        for i in range(len(per_class_history)):
            color = cmap(i % 10)
            plt.plot(epochs, per_class_history[i], color=color, linewidth=1.5, label=str(i))
        plt.title('Per-class Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
        timestamp2 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path2 = os.path.join(base_dir, f'per_class_accuracy_{timestamp2}.png')
        plt.tight_layout()
        plt.savefig(out_path2)
        plt.close()
        abs_out_path2 = os.path.abspath(out_path2)
        if os.path.exists(out_path2):
            print(f'Per-class accuracy plot saved to {abs_out_path2}')
        else:
            print(f'Warning: per-class accuracy plot not found after save at {abs_out_path2}')
        # also save per-class history as CSV for easier debugging
        try:
            import csv
            csv_path = os.path.join(base_dir, f'per_class_history_{timestamp2}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['epoch'] + [f'class_{i}' for i in range(10)]
                writer.writerow(header)
                for ei, e in enumerate(epochs, start=1):
                    row = [ei] + [per_class_history[i][ei-1] if ei-1 < len(per_class_history[i]) else '' for i in range(10)]
                    writer.writerow(row)
            print(f'Per-class history CSV saved to {os.path.abspath(csv_path)}')
        except Exception as e:
            print('Failed to save per-class history CSV:', e)
    except Exception as e:
        print('Failed to save per-class accuracy plot:', e)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    # prepare confusion counts
    num_classes = 10
    conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            # update confusion
            for t, p in zip(y.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                conf[int(t)][int(p)] += 1
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f" Test accuracy: {acc:.2f}%")
    # per-class accuracy
    per_class = []
    print('Per-class accuracy:')
    for i in range(num_classes):
        row = conf[i]
        total_i = sum(row)
        correct_i = row[i]
        acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0.0
        per_class.append(acc_i)
        print(f'  {i}: {acc_i:.2f}%  (n={total_i})')
    # show top confusions for diagnosis
    print('Top confusions (true -> predicted counts):')
    flat = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf[i][j] > 0:
                flat.append((conf[i][j], i, j))
    flat.sort(reverse=True)
    for cnt, t, p in flat[:10]:
        print(f'  {t} -> {p}: {cnt}')

    return acc, per_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='784,1000,500,250',
                        help='Comma-separated layer sizes, e.g. "784,1000,500,250"')
    parser.add_argument('--epochs', type=int, default=5, help='fine-tune epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='pretrain epochs per rbm')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    # parse layers string into list of ints and attach to args
    try:
        args.layers = list(map(int, args.layers.split(',')))
    except Exception:
        print('Invalid --layers format. Expected comma-separated ints, e.g. 784,1000,500,250')
        raise
    train(args)

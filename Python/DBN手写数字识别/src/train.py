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
        acc = test(dbn, test_loader, device)
        accuracies.append(acc)

    os.makedirs('models', exist_ok=True)
    dbn.save('models/dbn_mnist.pth')
    print('Model saved to models/dbn_mnist.pth')

    # 绘制并保存准确率-轮次图
    try:
        os.makedirs('models', exist_ok=True)
        epochs = list(range(1, len(accuracies)+1))
        plt.figure()
        plt.plot(epochs, accuracies, marker='o')
        plt.title('Test Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join('models', f'accuracy_{timestamp}.png')
        plt.savefig(out_path)
        print(f'Accuracy plot saved to {out_path}')
    except Exception as e:
        print('Failed to save accuracy plot:', e)

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

    return acc

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

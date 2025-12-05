import argparse
import os
import sys
import torch
# 说明：以下将 `src/` 目录加入 `sys.path`，以便在直接运行脚本时能够通过
# `from dbn.model import DBN` 导入仓库内的 dbn 包。
# train.py 位于 `src/model`，所以包根目录为其父目录（即 `src/` 上层）。
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dbn.model import DBN
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm

# 字体处理：优先使用仓库内捆绑的 `src/utils/msyh.ttc`（若存在）以支持中文绘图标签。
bundled_font = os.path.join(SRC_DIR, 'utils', 'msyh.ttc')
used_font = None
if os.path.exists(bundled_font):
    try:
        fm.fontManager.addfont(bundled_font)
        fam = fm.FontProperties(fname=bundled_font).get_name()
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [fam] + matplotlib.rcParams.get('font.sans-serif', [])
        matplotlib.rcParams['axes.unicode_minus'] = False
        used_font = fam
    except Exception:
        used_font = None

if used_font:
    print(f"使用捆绑字体显示中文标签：{used_font}")
else:
    print('未应用捆绑字体；使用系统字体（可能无法正确显示中文）。')
import datetime

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 训练主流程说明：
# 1) 加载数据集 -> 2) 初始化 DBN -> 3) 逐层无监督预训练（dbn.pretrain） ->
# 4) 将 RBM 权重迁移到 classifier（transfer_rbm_to_classifier） -> 5) 监督微调（fine-tune）

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('阶段：加载 MNIST 数据集...')
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # 将输出（图像/CSV/模型文件）保存到工程根目录下的 `model/` 文件夹
    # 说明：train.py 位于 `src/model`，工程根为 `src` 的父目录
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    base_dir = os.path.join(PROJECT_ROOT, 'model')

    layer_sizes = getattr(args, 'layers', [784, 500, 200, 50])
    print(f'阶段：初始化 DBN，层结构：{layer_sizes}')
    dbn = DBN(layer_sizes)
    dbn.to(device)

    # pretrain
    print('阶段：RBM 逐层预训练...')
    dbn.pretrain(train_loader, epochs=args.pretrain_epochs, lr=args.pretrain_lr, device=device)

    # transfer pretrained rbm weights into classifier to initialize fine-tuning
    try:
        print('阶段：将 RBM 权重迁移到分类器初始化...')
        dbn.transfer_rbm_to_classifier()
    except Exception as e:
        print('警告：RBM 权重迁移失败：', e)

    # 监督微调（fine-tune）分类器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dbn.classifier.parameters(), lr=args.lr)
    print('阶段：对分类器进行监督微调...')
    accuracies = []
    # 跟踪每类的准确率历史（10 个类别）
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
        print(f"阶段：微调 第{epoch+1}/{args.epochs}轮 损失={total_loss/n:.6f}")
        acc, per_class = test(dbn, test_loader, device)
        accuracies.append(acc)
        # 将每类准确率（百分比）追加到历史记录中
        for i in range(len(per_class_history)):
            per_class_history[i].append(per_class[i] if i < len(per_class) else 0.0)

    os.makedirs(base_dir, exist_ok=True)
    try:
        model_path = os.path.join(base_dir, 'dbn_mnist.pth')
        dbn.save(model_path)
        abs_model_path = os.path.abspath(model_path)
        if os.path.exists(model_path):
            print(f'模型已保存到 {abs_model_path}')
        else:
            print(f'警告：尝试保存模型，但在该路径未找到文件 {abs_model_path}')
    except Exception as e:
        print('保存模型失败：', e)

    # 绘制并保存准确率-轮次图
    try:
        os.makedirs(base_dir, exist_ok=True)
        epochs = list(range(1, len(accuracies)+1))
        plt.figure()
        # 绘图时不显示每轮的点标记，以免 epoch 较多时图像过于拥挤
        plt.plot(epochs, accuracies, linewidth=1.5)
        plt.title('每轮测试准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(base_dir, f'accuracy_{timestamp}.png')
        plt.savefig(out_path)
        plt.close()
        abs_out_path = os.path.abspath(out_path)
        if os.path.exists(out_path):
            print(f'准确率图已保存到 {abs_out_path}')
        else:
            print(f'警告：保存准确率图后未找到文件 {abs_out_path}')
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
        plt.title('各类准确率随轮次变化')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.grid(True)
        plt.legend(title='类别', bbox_to_anchor=(1.02, 1), loc='upper left')
        timestamp2 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path2 = os.path.join(base_dir, f'per_class_accuracy_{timestamp2}.png')
        plt.tight_layout()
        plt.savefig(out_path2)
        plt.close()
        abs_out_path2 = os.path.abspath(out_path2)
        if os.path.exists(out_path2):
            print(f'每类准确率图已保存到 {abs_out_path2}')
        else:
            print(f'警告：保存每类准确率图后未找到文件 {abs_out_path2}')
        # 同时保存每类历史为 CSV 便于调试和后续分析
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
            print(f'每类历史 CSV 已保存到 {os.path.abspath(csv_path)}')
        except Exception as e:
            print('保存每类历史 CSV 失败：', e)
    except Exception as e:
        print('Failed to save per-class accuracy plot:', e)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    # 准备混淆矩阵计数结构
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
            # 更新混淆矩阵计数
            for t, p in zip(y.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                conf[int(t)][int(p)] += 1
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f" 测试准确率：{acc:.2f}%")
    # 计算并打印每类准确率
    per_class = []
    print('每类准确率：')
    for i in range(num_classes):
        row = conf[i]
        total_i = sum(row)
        correct_i = row[i]
        acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0.0
        per_class.append(acc_i)
        print(f'  {i}: {acc_i:.2f}%  (样本数={total_i})')
    # 打印最常见的混淆项，便于诊断模型误分情况
    print('主要混淆项（真实 -> 预测：计数）：')
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
                        help='逗号分隔的层大小，例如："784,1000,500,250"')
    parser.add_argument('--epochs', type=int, default=5, help='微调时的训练轮数')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='每个 RBM 的预训练轮数')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='预训练学习率')
    parser.add_argument('--lr', type=float, default=1e-3, help='微调阶段的学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='训练/评估时的批大小')
    args = parser.parse_args()
    # parse layers string into list of ints and attach to args
    try:
        args.layers = list(map(int, args.layers.split(',')))
    except Exception:
        print('无效的 --layers 格式。期望逗号分隔的整数，例如：784,1000,500,250')
        raise
    train(args)

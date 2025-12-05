"""在 MNIST 测试集上评估已保存的 DBN 模型并绘制结果。

保存文件：
- model/eval_confusion_<timestamp>.png  （混淆矩阵热力图）
- model/eval_perclass_<timestamp>.png  （每类准确率柱状图）
- model/eval_predictions_<timestamp>.csv （可选的预测结果 CSV）

用法：
    python src/evaluate.py --model model/dbn_mnist.pth
"""
import os
import sys
import argparse
import datetime
import csv
import numpy as np
import torch
# matplotlib 在下面导入；字体注册会在解析 SRC_DIR 后进行
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
# 将 `src/` 加入 sys.path，以便本地包能被正确导入
# evaluate.py 位于 `src/evaluate`，因此包根目录为 `src` 的父目录
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# If a bundled font `src/utils/msyh.ttc` exists, register and prefer it for plots.
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
    print(f"Using bundled font for Chinese labels: {used_font}")
else:
    print('No bundled font applied; using system fonts (may not render CJK).')
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_model_checkpoint(path, device):
    # 若模型文件不存在，给出更清晰的错误信息（相对路径传入时常见）
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"模型文件未找到：{path!r}（当前工作目录：{os.getcwd()}）")
    ck = torch.load(path, map_location=device)
    layer_sizes = ck.get('layer_sizes', None)
    return ck, layer_sizes

def build_model_from_ck(ck, layer_sizes, device):
    # 局部导入以避免循环依赖问题
    from dbn.model import DBN
    if layer_sizes is None:
        # 后备默认结构
        layer_sizes = [784, 1000, 500, 250]
    model = DBN(layer_sizes)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate(model, device, batch_size=256, output_dir='model', save_csv=True):
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

    # 绘制混淆矩阵
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('Blues')
    im = ax1.imshow(conf, interpolation='nearest', cmap=cmap)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
        xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)),
        ylabel='真实标签', xlabel='预测标签', title=f'混淆矩阵 (准确率={acc:.2f}%)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 在矩阵单元格上标注数值
    thresh = conf.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax1.text(j, i, format(conf[i, j], 'd'), ha='center', va='center',
                     color='white' if conf[i, j] > thresh else 'black')
    conf_path = os.path.join(output_dir, f'eval_confusion_{timestamp}.png')
    fig1.tight_layout()
    fig1.savefig(conf_path)

    # 绘制每类准确率柱状图
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(range(num_classes), per_class)
    ax2.set_xticks(range(num_classes))
    ax2.set_xlabel('类别')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('各类准确率')
    # 将纵轴范围设为 97-100 以突出微小差异
    ax2.set_ylim(97, 100)
    # 在每个柱子上方标注其准确率（百分比）
    for bar, val in zip(bars, per_class):
        h = bar.get_height()
        # ensure the annotation is within the visible range: if h < 97 place above 97
        y_pos = max(h, 97) + 0.05
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
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

    print(f'评估完成。总体准确率：{acc:.2f}%')
    print(f'混淆矩阵已保存到：{conf_path}')
    print(f'每类准确率图已保存到：{perclass_path}')
    if csv_path:
        print(f'预测结果 CSV 已保存到：{csv_path}')

    return {'accuracy': acc, 'confusion_path': conf_path, 'perclass_path': perclass_path, 'csv_path': csv_path}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='已保存模型文件路径（.pth）。若省略脚本将尝试自动发现模型')
    parser.add_argument('--batch_size', type=int, default=256, help='评估时的 batch 大小')
    parser.add_argument('--output_dir', default='model', help='评估结果保存目录')
    parser.add_argument('--no_csv', action='store_true', help='不保存预测结果的 CSV 文件')
    args = parser.parse_args()

    # if model not provided, try to find it automatically
    if args.model is None:
        try:
            from utils.paths import find_model_path
            args.model = find_model_path()
        except Exception:
            args.model = None
    if args.model is None:
        parser.error('未提供模型路径且自动发现失败。请使用 --model 指定模型，或将模型放置在标准位置。')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck, layer_sizes = load_model_checkpoint(args.model, device)
    model = build_model_from_ck(ck, layer_sizes, device)
    evaluate(model, device, batch_size=args.batch_size, output_dir=args.output_dir, save_csv=not args.no_csv)

if __name__ == '__main__':
    main()

"""在 `sample_test/` 中对保存的模型进行评估（这些样本由 `extract_samples.py` 产生）。

用法：
    python src\eval_samples.py --model model/dbn_mnist.pth --samples sample_test

该脚本会加载 `samples/<digit>/` 目录下的每张图片，运行模型预测（不反色），
并生成包含预测结果的 CSV 以及一个简单的准确率报告。
"""
import os
import sys
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
    # 将 `src/` 加入 sys.path，以便在运行本脚本时能通过 `from dbn.model import DBN` 导入本地包
    SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    parser.add_argument('--model', default=None, help='已保存模型文件的路径（.pth）。若省略，脚本会在常见位置搜索')
    parser.add_argument('--samples', default='sample_test', help='包含待评估样本的目录，结构为 <digit>/<image>.png')
    parser.add_argument('--output', default=os.path.join('model', 'sample_eval.csv'), help='保存预测结果的 CSV 路径（默认放在 project_root/model/）')
    args = parser.parse_args()

    # 若未提供模型路径，则尝试自动解析
    if args.model is None:
        try:
            from utils.paths import find_model_path
            args.model = find_model_path()
        except Exception:
            args.model = None
    if args.model is None:
        parser.error('未提供模型路径且自动发现失败。请使用 --model 指定模型或将模型放在常见位置')

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
                print('打开图片失败：', path, e)
                continue
            # sample_test 中的图片是从 MNIST 原始数组保存的；不要反转颜色
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
    print(f'样本测试准确率：{acc:.2f}% ({correct}/{total})')
    print('各类准确率：')
    for k in sorted(per_class.keys(), key=lambda x:int(x)):
        stats = per_class[k]
        n = stats['n']
        tp = stats['tp']
        acc_k = 100.0*tp/n if n>0 else 0.0
        print(f' {k}: {acc_k:.2f}% ({tp}/{n})')
    print('预测结果已保存到：', args.output)

if __name__ == '__main__':
    main()

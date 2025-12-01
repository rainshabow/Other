# DBN 手写数字识别

这是一个演示项目，包含：
- 基于 PyTorch 的简单 RBM 实现和堆叠预训练（DBN 风格）
- 在 MNIST 上的预训练与微调训练脚本
- 一个基于 Tkinter 的 GUI，用于绘制或加载图片并进行预测

快速开始：

1. 创建并激活虚拟环境（Windows PowerShell）

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 训练模型（会自动下载 MNIST）

```powershell
python src\train.py --epochs 3
```

3. 运行 GUI

```powershell
python src\gui\app.py
```

说明：训练为了演示设置了较少的 epoch。若要高精度，请增加预训练和微调的 epoch。模型会被保存到 `models/dbn_mnist.pth`。

如果想在没有训练的情况下先试 GUI，可以更改 `src/gui/app.py` 中的 `MODEL_PATH` 为现有模型路径。

作者：示例代码，供学习和扩展使用。

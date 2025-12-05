import os
import sys
# 将 'src' 目录加入 sys.path，以便在从项目根运行脚本时能导入本地包 'dbn'
# 例如：`python src\gui\app.py`
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageOps, ImageGrab, ImageFilter, ImageTk, ImageDraw

# Pillow 10 移除了 Image.ANTIALIAS 常量，使用向后兼容的 RESAMPLE 变量
if hasattr(Image, 'Resampling'):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    # 兼容更老版本的 Pillow
    RESAMPLE = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', 1))
import numpy as np
import torch
from dbn.model import DBN

try:
    from utils.paths import find_model_path
except Exception:
    find_model_path = None

# 从常见位置解析模型路径（环境变量、项目目录等）。若未找到，
# 则回退到兼容的相对路径 `models/dbn_mnist.pth`，以保持行为不变。
MODEL_PATH = None
if find_model_path:
    try:
        MODEL_PATH = find_model_path()
    except Exception:
        MODEL_PATH = None
if not MODEL_PATH:
    # 若未找到模型，则默认使用项目根下的 `model/` 目录（单数）
    MODEL_PATH = os.path.join(ROOT, 'model', 'dbn_mnist.pth')

# 存放调试输出的目录（若模型所在目录可用则优先使用该目录）
DEBUG_DIR = (os.path.dirname(MODEL_PATH) if MODEL_PATH and os.path.dirname(MODEL_PATH) else os.path.join(ROOT, 'model'))

class App:
    def __init__(self, root):
        self.root = root
        root.title('DBN 手写数字识别 测试 GUI')

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<B1-Motion>", self.draw)
        # 支持多笔写字：在鼠标释放时重置起点，避免将不同笔画连接在一起
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)
        self.old_x = None
        self.old_y = None
        # maintain an in-memory PIL image to mirror canvas drawing (avoids screen grabs)
        self._pil_image = Image.new('L', (280, 280), 'white')
        self._pil_draw = ImageDraw.Draw(self._pil_image)

        # 使用一个固定宽度的按钮容器，使按钮总宽度不超过画布宽度(280px)，并让按钮等分占满容器
        self.style = ttk.Style()
        try:
            self.style.configure('Rounded.TButton', relief='flat', padding=6)
        except Exception:
            pass
        # 创建一个与画布等宽的按钮容器，并将按钮放入其中，使按钮能等分并自适应宽度
        self.button_frame = tk.Frame(root, width=280)
        self.button_frame.grid(row=1, column=0, columnspan=4, pady=4)
        # prevent the frame from shrinking to children so total width remains 280
        self.button_frame.grid_propagate(False)
        btn_padx = 2
        btn_pady = 2
        btn_style = 'Rounded.TButton'
        self.btn_clear = ttk.Button(self.button_frame, text='清除', command=self.clear, style=btn_style)
        self.btn_load = ttk.Button(self.button_frame, text='载入图片', command=self.load_image, style=btn_style)
        self.btn_pred = ttk.Button(self.button_frame, text='识别', command=self.predict, style=btn_style)
        for b in (self.btn_clear, self.btn_load, self.btn_pred):
            b.pack(side='left', expand=True, fill='x', padx=btn_padx, pady=btn_pady)

        # 右侧面板：上方显示调试用的预处理图片，下方显示预测结果
        self.side_frame = tk.Frame(root)
        self.side_frame.grid(row=0, column=4, rowspan=3, padx=8, pady=8, sticky='n')

        # 调试用图像显示区域（右侧面板顶部），初始为全黑占位图
        black = Image.new('L', (140, 140), 'black')
        self.debug_tkimg = ImageTk.PhotoImage(black)
        self.debug_img_label = tk.Label(self.side_frame, image=self.debug_tkimg, width=140, height=140, bg='black')
        self.debug_img_label.pack(padx=4, pady=(4, 8))

        # 结果容器（右侧面板底部） - 固定高度，结果文本垂直居中显示
        self.result_container = tk.Frame(self.side_frame, width=140, height=100)
        self.result_container.pack(fill='both', expand=False, padx=4, pady=4)
        self.result_container.pack_propagate(False)
        self.label = tk.Label(self.result_container, text='结果: -', font=('Arial', 16))
        # center vertically and horizontally
        self.label.pack(expand=True)

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # store last loaded original image (PIL Image, grayscale) and path
        self.loaded_image = None
        self.loaded_image_path = None
        if MODEL_PATH and os.path.exists(str(MODEL_PATH)):
            self.load_model(MODEL_PATH)
        else:
            messagebox.showinfo('提示', f'未找到模型文件: {MODEL_PATH}\n请先运行训练脚本 `python src\\model\\train.py` 或将模型放到项目根目录下的 `model/dbn_mnist.pth`')

    def draw(self, event):
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, x, y, width=18, fill='black', capstyle='round', smooth=True)
            # draw the same line onto the in-memory PIL image
            self._pil_draw.line([self.old_x, self.old_y, x, y], fill=0, width=18)
        self.old_x = x
        self.old_y = y

    def reset_stroke(self, event=None):
        """在鼠标释放时清除当前起点，使下一笔不会与上一笔相连。"""
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.canvas.delete('all')
        self.old_x = None
        self.old_y = None
        # clear any loaded original image as well
        self.loaded_image = None
        self.loaded_image_path = None
        # reset in-memory drawing
        self._pil_image = Image.new('L', (280, 280), 'white')
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self.label.config(text='结果: -')
        # reset debug image display to default black
        try:
            black = Image.new('L', (140, 140), 'black')
            self.debug_tkimg = ImageTk.PhotoImage(black)
            self.debug_img_label.config(image=self.debug_tkimg)
        except Exception:
            pass

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg')])
        if not path:
            return
        # 载入原始灰度图片（保持原始分辨率和极性）
        orig = Image.open(path).convert('L')

        # 先清空画布（但保留已载入的图片用于预测），以免 clear() 意外移除我们希望保留的 loaded_image
        self.clear()

        # 保存原始图片（不反色）以用于预测
        self.loaded_image = orig.copy()
        self.loaded_image_path = path

        # 为画布创建一个可显示的反色/缩放版本（保持之前的视觉行为）
        disp = ImageOps.invert(orig)
        disp = disp.resize((280, 280), RESAMPLE)
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tkimg)

    def get_canvas_image(self):
        # 返回内存中与画布同步的 PIL 图像副本
        return self._pil_image.copy()

    def preprocess(self, img, invert=True, eval_style_resize=False):
        """对 PIL 图像进行预处理以供模型输入。

        Args:
            img: PIL.Image (grayscale)
            invert: whether to invert colors (use True for canvas images, False for MNIST-style files)
            eval_style_resize: if True, use the same resize call as `eval_samples.py` (no explicit resample),
                               otherwise use high-quality `RESAMPLE` for canvas resizing.
        """
        # 若为画布图像（invert=True），执行类似 MNIST 的居中与缩放：裁剪墨迹 bbox、按比例缩放至
        # (size - 2*padding) 内、再填充到目标 size，最后反色。
        size = 28
        if invert and not eval_style_resize:
            img = self._center_and_scale(img, size=size, padding=4)
            img = ImageOps.invert(img)
        else:
            # 载入的图片：遵循 eval_samples 的行为或使用高质量 RESAMPLE
            if eval_style_resize:
                img = img.resize((size, size))
            else:
                img = img.resize((size, size), RESAMPLE)
            if invert:
                img = ImageOps.invert(img)

        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.reshape(1, -1)
        tensor = torch.from_numpy(arr).to(self.device)
        return tensor

    def _center_and_scale(self, img, size=28, padding=4):
        """Center and scale the ink in a PIL grayscale image to a square of given size.

        Works on images with black strokes (low values) on white background (high values).
        """
        arr = np.array(img)
        # consider pixels as ink if they are significantly darker than white
        mask = arr < 250
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            # nothing drawn; just resize to target
            return img.resize((size, size), RESAMPLE)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # PIL box is (left, upper, right, lower)
        crop = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        w, h = crop.size
        target = size - 2 * padding
        if target <= 0:
            target = size
        scale = min(target / float(w), target / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = crop.resize((new_w, new_h), RESAMPLE)
        new_img = Image.new('L', (size, size), 'white')
        paste_x = (size - new_w) // 2
        paste_y = (size - new_h) // 2
        new_img.paste(resized, (paste_x, paste_y))
        return new_img

    def load_model(self, path):
        # 尝试读取 checkpoint 以确定保存的层大小
        try:
            ck = torch.load(path, map_location=self.device)
            layer_sizes = ck.get('layer_sizes', [784, 500, 200, 50])
            model = DBN(layer_sizes)
            model.load_state_dict(ck['state_dict'])
        except Exception:
            # 回退：使用默认层大小构建模型并由 DBN.load 处理加载
            layer_sizes = [784, 500, 200, 50]
            model = DBN(layer_sizes)
            model.load(path, map_location=self.device)
        model.to(self.device)
        model.eval()
        self.model = model
        print('模型已加载:', path)

    def predict(self):
        if self.model is None:
            messagebox.showwarning('警告', '未加载模型，请先训练并保存模型到 model/dbn_mnist.pth')
            return
        # 优先使用原始载入的图片文件（若存在）；否则使用画布图像
        if self.loaded_image is not None:
            # loaded_image is expected to be a MNIST-style image (foreground=white)
            img = self.loaded_image
            tensor = self.preprocess(img, invert=False, eval_style_resize=True)
        else:
            # canvas image has black strokes on white background -> invert needed
            img = self.get_canvas_image()
            tensor = self.preprocess(img, invert=True, eval_style_resize=False)
        # 保存并打印原始输入的统计信息以便调试
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            # save original loaded image if present
            if self.loaded_image is not None:
                orig_arr = np.array(self.loaded_image).astype(np.float32)/255.0
                from PIL import Image as PILImage
                PILImage.fromarray((orig_arr*255).astype('uint8')).save(os.path.join(DEBUG_DIR, 'debug_original.png'))
                print(f"[DEBUG] original image stats min={orig_arr.min():.4f} max={orig_arr.max():.4f} mean={orig_arr.mean():.4f} std={orig_arr.std():.4f}")
            # save current canvas image
            canvas_img = self.get_canvas_image()
            canvas_arr = np.array(canvas_img).astype(np.float32)/255.0
            from PIL import Image as PILImage
            PILImage.fromarray((canvas_arr*255).astype('uint8')).save(os.path.join(DEBUG_DIR, 'debug_canvas.png'))
            print(f"[DEBUG] canvas image stats min={canvas_arr.min():.4f} max={canvas_arr.max():.4f} mean={canvas_arr.mean():.4f} std={canvas_arr.std():.4f}")
        except Exception as e:
            print('调试图片保存失败：', e)
        # --- Debugging: save preprocessed image and print tensor stats/probabilities ---
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            arr = tensor.cpu().numpy().reshape(28,28)
            # scale to 0-255 for human inspection
            from PIL import Image as PILImage
            img_debug = PILImage.fromarray((arr*255).astype('uint8'))
            debug_path = os.path.join(DEBUG_DIR, 'debug_preprocessed.png')
            img_debug.save(debug_path)
            print(f'[DEBUG] Preprocessed image saved to: {debug_path}')
            print(f'[DEBUG] tensor min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}')
            # update GUI debug image (resize for display)
            try:
                disp = img_debug.resize((140, 140))
                self.debug_tkimg = ImageTk.PhotoImage(disp)
                self.debug_img_label.config(image=self.debug_tkimg)
            except Exception as e:
                print('更新 GUI 调试图片失败：', e)
        except Exception as e:
            print('调试保存失败：', e)

        with torch.no_grad():
            out = self.model(tensor)
            # compute softmax probs
            probs = torch.softmax(out, dim=1).cpu().numpy().ravel()
            top3 = sorted([(p,i) for i,p in enumerate(probs)], reverse=True)[:3]
            pred = int(out.argmax(dim=1).item())
        print('[DEBUG] top 概率:', ', '.join([f'{i}:{p:.4f}' for p,i in top3]))
        self.label.config(text=f'结果: {pred}')

if __name__ == '__main__':
    try:
        from PIL import ImageTk
    except Exception:
        pass
    root = tk.Tk()
    app = App(root)
    root.mainloop()

import os

def _project_root(caller_file=__file__):
    # Project root is three levels up from src/utils/paths.py -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(caller_file))))


def find_model_path(preferred: str = None, candidates: list = None) -> str:
    """尝试定位已保存的 DBN 模型文件路径。

    搜索顺序：
    1. 若提供了 `preferred` 且路径存在（绝对或相对），直接返回它。
    2. 若提供了 `candidates` 列表，按顺序检查其中的路径（绝对或相对）。
    3. 在一组相对于项目根或当前工作目录的常见位置中查找。
    返回第一个存在的绝对路径，若未找到则返回 `None`。
    """
    # 1) 首先检查 preferred 参数
    if preferred:
        # 若为相对路径：先按给定路径检查，再尝试以项目根为基准的相对路径
        if os.path.isabs(preferred):
            if os.path.exists(preferred):
                return preferred
        else:
            # 先按给定相对路径检查
            if os.path.exists(preferred):
                return os.path.abspath(preferred)
            # 再尝试以项目根为基准的相对路径
            root = _project_root()
            p = os.path.join(root, preferred)
            if os.path.exists(p):
                return os.path.abspath(p)

    # 2) 检查传入的 candidates 列表
    if candidates:
        for c in candidates:
            if os.path.isabs(c):
                if os.path.exists(c):
                    return c
            else:
                if os.path.exists(c):
                    return os.path.abspath(c)
                root = _project_root()
                p = os.path.join(root, c)
                if os.path.exists(p):
                    return os.path.abspath(p)

    # 3) 默认的常见位置
    root = _project_root()
    # 优先检查项目根下的 `model/` 文件夹（单数），然后检查历史路径 `models/` 和 `src/model`
    common = [
        os.path.join(root, 'model', 'dbn_mnist.pth'),
        os.path.join(root, 'models', 'dbn_mnist.pth'),
        os.path.join(root, 'src', 'model', 'dbn_mnist.pth'),
        os.path.join(root, 'models', 'dbn_mnist.pt'),
        os.path.join(root, 'dbn_mnist.pth'),
    ]
    # 还检查当前工作目录下的一些相对文件名
    cwd_names = ['model/dbn_mnist.pth', 'models/dbn_mnist.pth', 'dbn_mnist.pth']

    for p in common + cwd_names:
        if os.path.exists(p):
            return os.path.abspath(p)

    return None

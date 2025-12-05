import os
import matplotlib.font_manager as fm


def find_chinese_font_name():
    """尝试在系统中查找支持中文的字体。

    返回适合作为 `matplotlib.rcParams['font.family']` 的字体家族名称，
    若未发现合适的中文字体则返回 `None`。
    """
    # 在 Windows 上常见的首选字体家族名
    candidates = [
        'Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun',
        'WenQuanYi Zen Hei', 'AR PL UMing CN', 'Noto Sans CJK SC', 'PingFang SC'
    ]

    # 首先尝试精确匹配字体家族名称
    for f in fm.fontManager.ttflist:
        try:
            if f.name in candidates:
                return f.name
        except Exception:
            continue

    # 其次在字体文件名中查找常见子串以作匹配
    substrs = ['yahei', 'simhei', 'simsun', 'wqy', 'noto', 'uming', 'pingfang', 'zh']
    for f in fm.fontManager.ttflist:
        try:
            fn = os.path.basename(f.fname).lower()
            for s in substrs:
                if s in fn:
                    return f.name
        except Exception:
            continue

    return None


def apply_chinese_font():
    """如果可用，给 matplotlib 应用支持中文的字体并返回所用字体名。

    返回已应用的字体名（字符串），若未应用任何特殊字体则返回 None。
    同时将 `axes.unicode_minus` 设为 False 以避免负号显示问题。
    """
    import matplotlib
    # 若本模块旁存在捆绑字体文件（msyh.ttc），优先注册并使用它
    bundled = os.path.join(os.path.dirname(__file__), 'msyh.ttc')
    if os.path.exists(bundled):
        try:
            fm.fontManager.addfont(bundled)
            # 重建 matplotlib 使用的字体列表（部分 matplotlib 版本会在 addfont 后自动刷新）
            fam = fm.FontProperties(fname=bundled).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [fam] + matplotlib.rcParams.get('font.sans-serif', [])
            matplotlib.rcParams['axes.unicode_minus'] = False
            return fam
        except Exception:
            # 注册失败则回退到自动发现
            pass

    name = find_chinese_font_name()
    if name:
        # 优先通过 sans-serif 列表设置家族名，这在多数后端中会被尊重
        try:
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [name] + matplotlib.rcParams.get('font.sans-serif', [])
        except Exception:
            matplotlib.rcParams['font.family'] = name
    # 确保负号能正常显示
    matplotlib.rcParams['axes.unicode_minus'] = False
    return name

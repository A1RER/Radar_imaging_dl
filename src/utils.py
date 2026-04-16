"""
utils.py
通用工具函数：dB转换、软阈值、图像质量指标、随机种子控制
"""
import torch
import numpy as np
import random


# ── 随机种子 ──────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    固定 random / numpy / torch 的随机种子，保证训练可复现。
    注意：cuDNN deterministic 模式会降低训练速度，仅在需要精确复现时启用。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── dB 转换 ──────────────────────────────────────────────────────────

def db20(x: torch.Tensor) -> torch.Tensor:
    """
    归一化幅度 → dB（对应 MATLAB 中的 db20）
    db = 20 * log10(|x| / max(|x|))
    """
    x_abs = torch.abs(x)
    max_val = x_abs.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return 20.0 * torch.log10(x_abs / max_val + 1e-10)


# ── 软阈值（复数） ────────────────────────────────────────────────────

def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """
    对实数非负张量做软阈值（对应 MATLAB ADMM 中的 z-update）
    输入 x 为幅度值（≥ 0），输出 max(x - threshold, 0)
    """
    return torch.clamp(x - threshold, min=0.0)


# ── 图像质量指标 ──────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    峰值信噪比（PSNR），在幅度域计算。
    pred / target: 任意形状，取幅度后按最大值归一化到 [0,1]。
    """
    pred_amp   = torch.abs(pred)
    target_amp = torch.abs(target)
    max_val    = target_amp.max().clamp(min=1e-10)
    pred_amp   = pred_amp   / max_val
    target_amp = target_amp / max_val
    mse = torch.mean((pred_amp - target_amp) ** 2).clamp(min=1e-10)
    return 20.0 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))


def imaging_entropy(image: torch.Tensor) -> torch.Tensor:
    """
    成像熵（越低代表图像越聚焦）。
    image: [..., H, W]，取幅度后计算。
    """
    amp = torch.abs(image)
    amp = amp / amp.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return -(amp * torch.log(amp + 1e-10)).sum(dim=(-2, -1))


# ── 数据归一化 ────────────────────────────────────────────────────────

def normalize_complex(x: torch.Tensor) -> torch.Tensor:
    """按幅度最大值归一化复数张量。"""
    max_amp = torch.abs(x).amax(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return x / max_amp


def to_numpy_amp_db(x: torch.Tensor, dynamic_range: float = 30.0) -> np.ndarray:
    """
    将 torch 复数张量转为归一化 dB 幅度 numpy 数组，供 matplotlib 显示。
    dynamic_range: 显示动态范围 (dB)，超出部分裁切为 -dynamic_range。
    """
    amp = torch.abs(x).detach().cpu()
    amp_db = 20.0 * torch.log10(amp / amp.max().clamp(min=1e-10) + 1e-10)
    return amp_db.clamp(min=-dynamic_range).numpy()

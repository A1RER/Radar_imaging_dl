"""
losses.py
成像重建任务的损失函数集合。

针对雷达成像的特点：
- MSE    ：抑制整体幅度误差
- L1     ：促进稀疏性（雷达图像本就是稀疏点散射集合）
- SSIM   ：保持结构相似性，对降噪任务尤其关键（相比 MSE 更接近人眼感知）
- Combined：以上三者的加权组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SSIM 实现（在幅度域计算） ────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """生成 2D 高斯核，形状 [1, 1, window_size, window_size]。"""
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma:       float = 1.5,
    data_range:  float = 1.0,
) -> torch.Tensor:
    """
    计算两张图的结构相似性指数（SSIM），返回 [0, 1] 之间的标量。
    x, y: [B, nx, nkr] 实数，已归一化到 [0, data_range]。
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)   # [B, 1, H, W]
        y = y.unsqueeze(1)

    kernel = _gaussian_kernel(window_size, sigma, x.device, x.dtype)
    pad    = window_size // 2

    mu_x  = F.conv2d(x, kernel, padding=pad)
    mu_y  = F.conv2d(y, kernel, padding=pad)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x * x, kernel, padding=pad) - mu_xx
    sigma_yy = F.conv2d(y * y, kernel, padding=pad) - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2))
    return ssim_map.mean()


# ── 组合损失 ──────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L = w_mse * MSE + w_l1 * L1 + w_ssim * (1 - SSIM)

    所有分量都在幅度域（实数）计算。网络输出和标签均假定为非负实数。
    """

    def __init__(
        self,
        w_mse:  float = 1.0,
        w_l1:   float = 1e-4,
        w_ssim: float = 0.1,
    ):
        super().__init__()
        self.w_mse  = w_mse
        self.w_l1   = w_l1
        self.w_ssim = w_ssim

    def forward(
        self,
        pred:   torch.Tensor,   # [B, nx, nkr] real
        target: torch.Tensor,   # [B, nx, nkr] real
    ) -> torch.Tensor:
        # 归一化到 [0, 1]，SSIM 需要已知动态范围
        max_val   = target.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
        pred_n    = pred   / max_val
        target_n  = target / max_val

        loss_mse  = F.mse_loss(pred_n, target_n)
        loss_l1   = pred_n.abs().mean()
        loss_ssim = 1.0 - ssim(pred_n, target_n, data_range=1.0)

        return self.w_mse * loss_mse + self.w_l1 * loss_l1 + self.w_ssim * loss_ssim

"""
model.py
深度展开 Consensus-ADMM 网络（Deep Unrolling CADMM）

对应 MATLAB 的 MMV_l1_l2_cadmmfor3.m，将每次 ADMM 迭代展开为一个可学习的网络层：
  - 每层独立的可学习参数：λ_k（稀疏正则强度）、ρ_k（惩罚系数）、thresh_k（一致性阈值）
  - 感知矩阵 A（FFT 矩阵）固定，由物理模型决定，不参与训练
  - 前向传播等价于 K 次 ADMM 迭代，但参数通过端到端训练优化

MATLAB ADMM 对照：
  x-update : xt_i = (A'A + ρI)^{-1} (A'y_i - o_i + ρz)
  z-update : a_i  = xt_i + o_i/ρ
             mask = ∏_i [db20(a_i) > thresh]
             z    = soft_threshold(mask * |mean(a_i)|, λ/ρ)
  o-update : o_i  = o_i + ρ(xt_i - z)
"""
import torch
import torch.nn as nn
import math
from .utils import db20, soft_threshold


class CADMMLayer(nn.Module):
    """
    单次展开的 Consensus-ADMM 迭代。

    Parameters
    ----------
    n_apertures : int
        子孔径数量（默认 3，对应 MATLAB 的 3 个目标/快拍）
    init_rho    : float
        ρ 初始值（对应 MATLAB 的 400）
    init_lam    : float
        λ/ρ 初始值，即软阈值初始大小（对应 MATLAB 的 60000/400 = 150）
    init_thresh : float
        一致性判决 dB 阈值初始值（对应 MATLAB 的 -25 dB）
    """

    def __init__(
        self,
        n_apertures: int   = 3,
        init_rho:    float = 400.0,
        init_lam:    float = 150.0,
        init_thresh: float = -25.0,
    ):
        super().__init__()
        self.n_apertures = n_apertures

        # 用 log 参数化保证正值
        self.log_rho    = nn.Parameter(torch.tensor(math.log(init_rho)))
        self.log_lam    = nn.Parameter(torch.tensor(math.log(init_lam)))
        # 阈值直接学习（可正可负）
        self.thresh_db  = nn.Parameter(torch.tensor(init_thresh))

    @property
    def rho(self) -> torch.Tensor:
        return self.log_rho.exp()

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lam.exp()

    def forward(
        self,
        y_list:   list,   # list of [B, nkr, nkx] complex
        A:        torch.Tensor,  # [nkx, nx] complex
        z:        torch.Tensor,  # [B, nx, nkr] real
        o_list:   list,   # list of [B, nx, nkr] complex
    ):
        """
        Parameters
        ----------
        y_list  : 长度为 n_apertures 的列表，每个元素 [B, nkr, nkx] complex
        A       : [nkx, nx] complex，感知矩阵
        z       : [B, nx, nkr] real，共享图像变量
        o_list  : 长度为 n_apertures 的列表，每个元素 [B, nx, nkr] complex，对偶变量

        Returns
        -------
        z_new    : [B, nx, nkr] real
        o_list_new : list of [B, nx, nkr] complex
        """
        rho = self.rho
        lam = self.lam

        # ── 预计算 (A^H A + ρI)^{-1} ──────────────────────────────────
        # A: [nkx, nx] → A^H A: [nx, nx]
        AHA = A.conj().T @ A                                      # [nx, nx]
        M   = AHA + rho * torch.eye(AHA.shape[0], dtype=AHA.dtype, device=AHA.device)

        # ── x-update（每个子孔径独立） ────────────────────────────────
        xt_list = []
        for y, o in zip(y_list, o_list):
            # y: [B, nkr, nkx] → y^T: [B, nkx, nkr]
            y_t  = y.permute(0, 2, 1)                             # [B, nkx, nkr]
            # A^H y_t: [nx, nkx] @ [B, nkx, nkr] = [B, nx, nkr]
            AHy  = torch.einsum('ij,bjk->bik', A.conj().T, y_t)  # [B, nx, nkr]
            rhs  = AHy - o + rho * z.to(AHy.dtype)               # [B, nx, nkr]
            # torch.linalg.solve 广播求解 M @ xt = rhs
            xt   = torch.linalg.solve(M, rhs)                    # [B, nx, nkr]
            xt_list.append(xt)

        # ── z-update（一致性 + 软阈值）────────────────────────────────
        a_list = [xt + o / rho for xt, o in zip(xt_list, o_list)]

        # 一致性 mask：用 sigmoid 软化替代硬判决，使梯度可通过
        # sigmoid(10 * (db20(a) - thresh)) ≈ step(db20(a) > thresh)
        mask = torch.ones_like(torch.abs(a_list[0]))
        for a in a_list:
            mask = mask * torch.sigmoid(10.0 * (db20(a) - self.thresh_db))

        a_mean = sum(torch.abs(a) for a in a_list) / self.n_apertures  # [B, nx, nkr]
        a_gated = mask * a_mean                                         # [B, nx, nkr]

        z_new = soft_threshold(a_gated, lam / rho)                     # [B, nx, nkr] real

        # ── o-update（对偶变量）──────────────────────────────────────
        o_list_new = [
            o + rho * (xt - z_new.to(xt.dtype))
            for o, xt in zip(o_list, xt_list)
        ]

        return z_new, o_list_new


class CADMMNet(nn.Module):
    """
    深度展开 Consensus-ADMM 网络。

    将 K 个 CADMMLayer 串联，每层参数独立可学习。

    Parameters
    ----------
    K           : int   展开层数（对应 ADMM 迭代次数，建议 5~15）
    n_apertures : int   子孔径数量（默认 3）
    """

    def __init__(self, K: int = 10, n_apertures: int = 3):
        super().__init__()
        self.K = K
        self.n_apertures = n_apertures
        self.layers = nn.ModuleList([
            CADMMLayer(n_apertures=n_apertures) for _ in range(K)
        ])

    def forward(
        self,
        sig:  torch.Tensor,   # [B, 3, nkr, nkx] complex
        A:    torch.Tensor,   # [nkx, nx] complex
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sig : [B, 3, nkr, nkx] complex — 3 个子孔径距离压缩数据
        A   : [nkx, nx] complex — 感知矩阵（FFT 矩阵，物理固定）

        Returns
        -------
        z : [B, nx, nkr] float — 重建图像幅度
        """
        B  = sig.shape[0]
        nx = A.shape[1]
        # nkr 从 sig 推算
        nkr = sig.shape[2]

        # 拆分子孔径
        y_list = [sig[:, i, :, :] for i in range(self.n_apertures)]  # 各 [B, nkr, nkx]

        # 初始化状态变量
        device, dtype = sig.device, sig.dtype
        z      = torch.zeros(B, nx, nkr, device=device, dtype=torch.float32)
        o_list = [
            torch.zeros(B, nx, nkr, device=device, dtype=dtype)
            for _ in range(self.n_apertures)
        ]

        # 前向展开 K 层
        for layer in self.layers:
            z, o_list = layer(y_list, A, z, o_list)

        return z   # [B, nx, nkr]

    def count_parameters(self) -> int:
        """返回可学习参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

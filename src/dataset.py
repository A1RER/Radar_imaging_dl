"""
dataset.py
加载由 matlab/generate_data.m 生成的 .mat 训练/测试数据。

.mat 文件结构（由 generate_data.m 保存）：
  tr_sig / te_sig  [N, 3, nkr, nkx]  complex64  - 3个子孔径输入
  tr_z   / te_z    [N, nx,  nkr]     float32    - 参考图像（标签）
  A                [nkx, nx]         complex64  - 感知矩阵
  kx               [1,   nkx]        float32    - 方位波数向量
"""
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import numpy as np


class RadarDataset(Dataset):
    """
    Parameters
    ----------
    mat_path     : str
        .mat 文件路径（train.mat 或 test.mat）
    sig_key      : str
        信号变量名，train.mat 用 'tr_sig'，test.mat 用 'te_sig'
    z_key        : str
        标签变量名，train.mat 用 'tr_z'，test.mat 用 'te_z'
    augment_snr  : tuple or None
        在线加噪增强。若给 (snr_min, snr_max)，则每次 __getitem__ 都额外添加
        该范围内随机 SNR 的高斯白噪声；None 表示不增强。训练集建议 (5, 25)，
        测试集应为 None 以保证指标可复现。
    """

    def __init__(
        self,
        mat_path:    str,
        sig_key:     str = 'tr_sig',
        z_key:       str = 'tr_z',
        augment_snr: Optional[Tuple[float, float]] = None,
    ):
        data = sio.loadmat(mat_path)

        # 信号输入：[N, 3, nkr, nkx] complex
        sig_np = data[sig_key].astype(np.complex64)
        self.sig = torch.from_numpy(sig_np)   # torch.complex64

        # 参考图像：[N, nx, nkr] float
        z_np = data[z_key].astype(np.float32)
        self.z_clean = torch.from_numpy(z_np)

        # 感知矩阵：[nkx, nx] complex（所有样本共享，存为属性）
        self.A  = torch.from_numpy(data['A'].astype(np.complex64))
        self.kx = torch.from_numpy(data['kx'].astype(np.float32)).squeeze()

        self.augment_snr = augment_snr

    def __len__(self) -> int:
        return self.sig.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        sig    : [3, nkr, nkx] complex64  - 3个子孔径输入
        z_clean: [nx,  nkr]   float32    - 无噪声参考图像
        """
        sig = self.sig[idx]

        # 在线加噪增强：在已有噪声基础上再加一层随机 SNR 高斯噪声，
        # 强迫网络见到更多噪声模式，提升抗噪鲁棒性
        if self.augment_snr is not None:
            snr_low, snr_high = self.augment_snr
            snr_db = torch.empty(1).uniform_(snr_low, snr_high).item()
            sig    = _add_complex_awgn(sig, snr_db)

        return sig, self.z_clean[idx]


def _add_complex_awgn(sig: torch.Tensor, snr_db: float) -> torch.Tensor:
    """向复数张量添加高斯白噪声，按指定 SNR (dB) 控制噪声功率。"""
    p_sig   = sig.abs().pow(2).mean()
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    std     = (p_noise / 2.0).sqrt()
    noise   = torch.complex(
        torch.randn_like(sig.real) * std,
        torch.randn_like(sig.imag) * std,
    )
    return sig + noise


def build_loaders(
    train_mat:      str,
    test_mat:       str,
    batch_size:     int                                  = 8,
    val_ratio:      float                                = 0.1,
    num_workers:    int                                  = 0,
    train_augment:  Optional[Tuple[float, float]]        = None,
):
    """
    构建训练、验证、测试 DataLoader。

    Parameters
    ----------
    train_mat     : data/train.mat 路径
    test_mat      : data/test.mat  路径
    batch_size    : 批大小
    val_ratio     : 从训练集中分出的验证比例
    num_workers   : DataLoader 工作进程数（Windows 建议设为 0）
    train_augment : 训练集在线加噪 SNR 范围 (low, high)，None 关闭增强

    Returns
    -------
    train_loader, val_loader, test_loader, A, kx
        A  : [nkx, nx] complex64 感知矩阵
        kx : [nkx]     float32  波数向量
    """
    train_full = RadarDataset(train_mat, sig_key='tr_sig', z_key='tr_z',
                              augment_snr=train_augment)
    test_ds    = RadarDataset(test_mat,  sig_key='te_sig', z_key='te_z',
                              augment_snr=None)  # 测试集不增强

    n_val   = max(1, int(len(train_full) * val_ratio))
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_full.A, train_full.kx

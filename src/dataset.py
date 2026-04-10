"""
dataset.py
加载由 matlab/generate_data.m 生成的 .mat 训练/测试数据。

.mat 文件结构（由 generate_data.m 保存）：
  tr_sig / te_sig  [N, 3, nkr, nkx]  complex64  - 3个子孔径输入
  tr_z   / te_z    [N, nx,  nkr]     float32    - 参考图像（标签）
  A                [nkx, nx]         complex64  - 感知矩阵
  kx               [1,   nkx]        float32    - 方位波数向量
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import numpy as np


class RadarDataset(Dataset):
    """
    Parameters
    ----------
    mat_path : str
        .mat 文件路径（train.mat 或 test.mat）
    sig_key  : str
        信号变量名，train.mat 用 'tr_sig'，test.mat 用 'te_sig'
    z_key    : str
        标签变量名，train.mat 用 'tr_z'，test.mat 用 'te_z'
    """

    def __init__(self, mat_path: str, sig_key: str = 'tr_sig', z_key: str = 'tr_z'):
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

    def __len__(self) -> int:
        return self.sig.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        sig    : [3, nkr, nkx] complex64  - 3个子孔径输入
        z_clean: [nx,  nkr]   float32    - 无噪声参考图像
        """
        return self.sig[idx], self.z_clean[idx]


def build_loaders(
    train_mat: str,
    test_mat:  str,
    batch_size: int   = 8,
    val_ratio:  float = 0.1,
    num_workers: int  = 0,
):
    """
    构建训练、验证、测试 DataLoader。

    Parameters
    ----------
    train_mat   : data/train.mat 路径
    test_mat    : data/test.mat  路径
    batch_size  : 批大小
    val_ratio   : 从训练集中分出的验证比例
    num_workers : DataLoader 工作进程数（Windows 建议设为 0）

    Returns
    -------
    train_loader, val_loader, test_loader, A, kx
        A  : [nkx, nx] complex64 感知矩阵
        kx : [nkx]     float32  波数向量
    """
    train_full = RadarDataset(train_mat, sig_key='tr_sig', z_key='tr_z')
    test_ds    = RadarDataset(test_mat,  sig_key='te_sig', z_key='te_z')

    n_val   = max(1, int(len(train_full) * val_ratio))
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, train_full.A, train_full.kx

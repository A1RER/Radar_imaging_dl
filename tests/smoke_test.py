"""
smoke_test.py
快速验证模型、损失、数据流形状是否正确。无需真实 .mat 数据，用随机张量跑通。

用法：
    python -m tests.smoke_test
"""
import sys
# Windows 默认 GBK 控制台无法打印 Unicode 符号，强制 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch

from src.model  import CADMMNet, CADMMLayer
from src.losses import CombinedLoss, ssim
from src.utils  import seed_everything, psnr, imaging_entropy


# ── 测试形状参数 ──────────────────────────────────────────────────────
B    = 2     # batch size
N_AP = 3     # 子孔径数
NKR  = 32    # 距离向像素
NKX  = 40    # 方位波数维度
NX   = 40    # 图像方位维度
K    = 5     # 展开层数


def test_single_layer():
    print('[1/4] 测试单层 CADMMLayer...')
    layer = CADMMLayer(n_apertures=N_AP)
    A = torch.randn(NKX, NX, dtype=torch.complex64)
    y_list = [torch.randn(B, NKR, NKX, dtype=torch.complex64) for _ in range(N_AP)]
    z = torch.zeros(B, NX, NKR, dtype=torch.float32)
    o_list = [torch.zeros(B, NX, NKR, dtype=torch.complex64) for _ in range(N_AP)]

    z_new, o_new = layer(y_list, A, z, o_list)
    assert z_new.shape  == (B, NX, NKR), f'z 形状错: {z_new.shape}'
    assert z_new.dtype  == torch.float32
    assert len(o_new)   == N_AP
    assert o_new[0].shape == (B, NX, NKR)
    assert o_new[0].dtype == torch.complex64
    print(f'    ✓ 输入 y[{B},{NKR},{NKX}] × A[{NKX},{NX}] → 输出 z[{B},{NX},{NKR}]')


def test_full_network():
    print('[2/4] 测试完整 CADMMNet 前向 + 反向...')
    net = CADMMNet(K=K, n_apertures=N_AP)
    A   = torch.randn(NKX, NX, dtype=torch.complex64)
    sig = torch.randn(B, N_AP, NKR, NKX, dtype=torch.complex64)
    target = torch.rand(B, NX, NKR, dtype=torch.float32)

    # forward
    z_pred = net(sig, A)
    assert z_pred.shape == (B, NX, NKR)
    assert z_pred.dtype == torch.float32
    print(f'    ✓ forward: sig[{B},{N_AP},{NKR},{NKX}] → z[{B},{NX},{NKR}]')

    # backward
    loss_fn = CombinedLoss()
    loss = loss_fn(z_pred, target)
    assert torch.isfinite(loss), f'损失非有限值: {loss}'
    loss.backward()

    # 检查每层参数都收到了梯度
    for i, layer in enumerate(net.layers):
        assert layer.log_rho.grad   is not None, f'layer{i}.log_rho 无梯度'
        assert layer.log_lam.grad   is not None, f'layer{i}.log_lam 无梯度'
        assert layer.thresh_db.grad is not None, f'layer{i}.thresh_db 无梯度'
    print(f'    ✓ backward: loss={loss.item():.4f}，{net.count_parameters()} 个可学习参数全部收到梯度')


def test_losses():
    print('[3/4] 测试损失函数...')
    pred   = torch.rand(B, NX, NKR)
    target = torch.rand(B, NX, NKR)

    # SSIM
    s = ssim(pred, target)
    assert -1.0 <= s.item() <= 1.0, f'SSIM 越界: {s.item()}'

    # 自损失（pred=target）应接近 0
    loss_fn = CombinedLoss()
    loss_self = loss_fn(target, target)
    assert loss_self.item() < 1e-3, f'自损失过大: {loss_self.item()}'
    print(f'    ✓ SSIM={s.item():.4f}，自损失={loss_self.item():.6f}')


def test_metrics():
    print('[4/4] 测试评价指标...')
    pred   = torch.rand(B, NX, NKR)
    target = torch.rand(B, NX, NKR)
    p = psnr(pred, target)
    e = imaging_entropy(pred)
    assert torch.isfinite(p)
    assert torch.isfinite(e).all()
    print(f'    ✓ PSNR={p.item():.2f} dB，imaging_entropy={e.mean().item():.4f}')


def main():
    seed_everything(42)
    print('=' * 60)
    print('Smoke Test — 验证 CADMMNet 数据流')
    print('=' * 60)
    try:
        test_single_layer()
        test_full_network()
        test_losses()
        test_metrics()
    except AssertionError as e:
        print(f'\n✗ 测试失败: {e}')
        sys.exit(1)
    print('=' * 60)
    print('全部通过 ✓')


if __name__ == '__main__':
    main()

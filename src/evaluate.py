"""
evaluate.py
加载训练好的模型，与传统 CADMM 对比，输出指标和可视化图像。

用法：
    python -m src.evaluate                          # 使用默认路径
    python -m src.evaluate --ckpt checkpoints/best.pth
"""
import argparse
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 支持中文显示（Windows/Linux 自动回退到系统已有字体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from .dataset import RadarDataset, _add_complex_awgn
from .model   import CADMMNet
from .utils   import psnr, imaging_entropy, to_numpy_amp_db


# ── 传统 CADMM 基线（对应 MATLAB MMV_l1_l2_cadmmfor3） ────────────────

def traditional_cadmm(
    sig:     torch.Tensor,   # [3, nkr, nkx] complex
    A:       torch.Tensor,   # [nkx, nx] complex
    max_iter: int   = 30,
    lamda:   float = 60000.0,
    rho:     float = 400.0,
) -> torch.Tensor:
    """
    在 PyTorch 中复现传统 CADMM，与网络输出做公平对比。
    输出：[nx, nkr] float（图像幅度）
    """
    n_ap = sig.shape[0]
    nx   = A.shape[1]
    nkr  = sig.shape[1]
    u    = lamda / rho

    AHA   = A.conj().T @ A
    M     = AHA + rho * torch.eye(nx, dtype=AHA.dtype, device=AHA.device)
    y_list = [sig[i].unsqueeze(0) for i in range(n_ap)]   # 各 [1, nkr, nkx]
    AHy_list = [
        torch.einsum('ij,bjk->bik', A.conj().T, y.permute(0, 2, 1))
        for y in y_list
    ]  # 各 [1, nx, nkr]

    z      = torch.zeros(1, nx, nkr, dtype=torch.float32, device=A.device)
    o_list = [torch.zeros(1, nx, nkr, dtype=A.dtype, device=A.device) for _ in range(n_ap)]

    for _ in range(max_iter):
        xt_list = [
            torch.linalg.solve(M, AHy - o + rho * z.to(A.dtype))
            for AHy, o in zip(AHy_list, o_list)
        ]
        a_list = [xt + o / rho for xt, o in zip(xt_list, o_list)]

        # 硬判决一致性 mask（-25 dB）
        def db20_np(x):
            x_abs = torch.abs(x)
            return 20.0 * torch.log10(x_abs / x_abs.amax().clamp(1e-10) + 1e-10)

        mask = torch.ones(1, nx, nkr, device=A.device)
        for a in a_list:
            mask = mask * (db20_np(a) > -25).float()

        a_mean = sum(torch.abs(a) for a in a_list) / n_ap
        z = torch.clamp(mask * a_mean - u, min=0.0)

        o_list = [o + rho * (xt - z.to(xt.dtype)) for o, xt in zip(o_list, xt_list)]

    return z.squeeze(0)   # [nx, nkr]


# ── 可视化：三列对比图（参考 / CADMM / 网络） ────────────────────────

def plot_comparison(z_clean, z_cadmm, z_net, save_path=None):
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 3, wspace=0.05)

    titles  = ['参考图像（无噪声）', '传统 CADMM', '深度展开网络（本文）']
    images  = [z_clean, z_cadmm, z_net]
    axes    = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        img_db = to_numpy_amp_db(img if isinstance(img, torch.Tensor)
                                 else torch.from_numpy(img.astype(np.float32)),
                                 dynamic_range=30)
        im = ax.imshow(img_db, cmap='jet', vmin=-30, vmax=0,
                       aspect='auto', origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('方位向')
        if ax == axes[0]:
            ax.set_ylabel('距离向')
        else:
            ax.set_yticks([])

    plt.colorbar(im, ax=axes[-1], label='归一化幅度 (dB)', fraction=0.046, pad=0.04)
    plt.suptitle('成像质量对比', fontsize=13, y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'图像已保存：{save_path}')
    plt.show()


# ── SNR 鲁棒性扫描 ────────────────────────────────────────────────────

def snr_sweep(model, test_ds, A, snr_list, n_samples, device):
    """
    对测试集每个样本在多个 SNR 下加噪，比较传统 CADMM 与网络的 PSNR。

    Returns
    -------
    dict: {'snr': [...], 'cadmm_psnr': [...], 'net_psnr': [...],
           'cadmm_ent': [...], 'net_ent': [...]}
    """
    n_eval = min(n_samples, len(test_ds))
    results = {
        'snr':        list(snr_list),
        'cadmm_psnr': [],
        'net_psnr':   [],
        'cadmm_ent':  [],
        'net_ent':    [],
    }

    for snr_db in snr_list:
        cadmm_p, net_p, cadmm_e, net_e = [], [], [], []
        for i in range(n_eval):
            sig, z_clean = test_ds[i]
            sig     = _add_complex_awgn(sig.to(device), snr_db)
            z_clean = z_clean.to(device)

            with torch.no_grad():
                z_cadmm = traditional_cadmm(sig, A)
                z_net   = model(sig.unsqueeze(0), A).squeeze(0)

            cadmm_p.append(psnr(z_cadmm, z_clean).item())
            net_p.append(  psnr(z_net,   z_clean).item())
            cadmm_e.append(imaging_entropy(z_cadmm).item())
            net_e.append(  imaging_entropy(z_net).item())

        results['cadmm_psnr'].append(np.mean(cadmm_p))
        results['net_psnr'].append(  np.mean(net_p))
        results['cadmm_ent'].append( np.mean(cadmm_e))
        results['net_ent'].append(   np.mean(net_e))
        print(f'SNR={snr_db:>5.1f} dB  '
              f'CADMM PSNR={results["cadmm_psnr"][-1]:>6.2f}  '
              f'网络 PSNR={results["net_psnr"][-1]:>6.2f}  '
              f'增益={results["net_psnr"][-1]-results["cadmm_psnr"][-1]:+.2f} dB')

    return results


def plot_snr_curve(results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    snrs = results['snr']

    axes[0].plot(snrs, results['cadmm_psnr'], 'o--', label='传统 CADMM', color='#888')
    axes[0].plot(snrs, results['net_psnr'],   's-',  label='深度展开网络', color='#d62728')
    axes[0].set_xlabel('输入 SNR (dB)')
    axes[0].set_ylabel('重建 PSNR (dB)')
    axes[0].set_title('不同噪声水平下的重建质量')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(snrs, results['cadmm_ent'], 'o--', label='传统 CADMM', color='#888')
    axes[1].plot(snrs, results['net_ent'],   's-',  label='深度展开网络', color='#d62728')
    axes[1].set_xlabel('输入 SNR (dB)')
    axes[1].set_ylabel('成像熵 (越低越聚焦)')
    axes[1].set_title('不同噪声水平下的图像聚焦度')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'SNR 曲线已保存：{save_path}')
    plt.show()


# ── 主评估流程 ────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    test_ds = RadarDataset(args.test_mat, sig_key='te_sig', z_key='te_z')
    A = test_ds.A.to(device)

    # 加载模型
    ckpt  = torch.load(args.ckpt, map_location=device)
    model = CADMMNet(K=ckpt['K'], n_apertures=3).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'已加载模型：{args.ckpt}（K={ckpt["K"]} 层，'
          f'验证 PSNR={ckpt["val_psnr"]:.2f} dB）')

    # SNR 鲁棒性扫描模式（不画单张对比图，直接跑曲线）
    if args.snr_sweep:
        snr_list = np.arange(args.snr_min, args.snr_max + 1e-6, args.snr_step).tolist()
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'\n── SNR 鲁棒性扫描：{args.snr_min}–{args.snr_max} dB '
              f'（步长 {args.snr_step}），每点 {args.n_samples} 样本 ──')
        results = snr_sweep(model, test_ds, A, snr_list, args.n_samples, device)
        plot_snr_curve(results, save_path=os.path.join(args.output_dir, 'snr_curve.png'))
        np.savez(os.path.join(args.output_dir, 'snr_curve.npz'), **results)
        print(f'原始数据已保存：{os.path.join(args.output_dir, "snr_curve.npz")}')
        return

    # 批量评估
    cadmm_psnr_list, net_psnr_list   = [], []
    cadmm_ent_list,  net_ent_list    = [], []
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(min(args.n_samples, len(test_ds))):
        sig, z_clean = test_ds[i]
        sig     = sig.to(device)
        z_clean = z_clean.to(device)

        # 传统 CADMM
        with torch.no_grad():
            z_cadmm = traditional_cadmm(sig, A)
            z_net   = model(sig.unsqueeze(0), A).squeeze(0)

        cadmm_psnr_list.append(psnr(z_cadmm, z_clean).item())
        net_psnr_list.append(  psnr(z_net,   z_clean).item())
        cadmm_ent_list.append( imaging_entropy(z_cadmm).item())
        net_ent_list.append(   imaging_entropy(z_net).item())

        # 保存前 args.save_images 张对比图
        if i < args.save_images:
            plot_comparison(
                z_clean.cpu(), z_cadmm.cpu(), z_net.cpu(),
                save_path=os.path.join(args.output_dir, f'compare_{i:03d}.png')
            )

    # 汇总
    print('\n── 评估结果汇总 ──────────────────────────────────────')
    print(f'{"指标":<20} {"传统 CADMM":>12} {"深度展开网络":>12}')
    print(f'{"PSNR (dB)↑":<20} {np.mean(cadmm_psnr_list):>12.2f} {np.mean(net_psnr_list):>12.2f}')
    print(f'{"成像熵↓":<20} {np.mean(cadmm_ent_list):>12.4f} {np.mean(net_ent_list):>12.4f}')


# ── CLI 入口 ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='评估深度展开 CADMM 网络')
    p.add_argument('--ckpt',        default='checkpoints/best.pth')
    p.add_argument('--test_mat',    default='data/test.mat')
    p.add_argument('--n_samples',   type=int, default=50,  help='评估样本数')
    p.add_argument('--save_images', type=int, default=5,   help='保存对比图数量')
    p.add_argument('--output_dir',  default='results')
    # SNR 鲁棒性扫描
    p.add_argument('--snr_sweep',   action='store_true',
                   help='启用 SNR 鲁棒性扫描模式（生成 PSNR-SNR 曲线）')
    p.add_argument('--snr_min',     type=float, default=0.0,  help='扫描起始 SNR (dB)')
    p.add_argument('--snr_max',     type=float, default=30.0, help='扫描终止 SNR (dB)')
    p.add_argument('--snr_step',    type=float, default=5.0,  help='扫描步长 (dB)')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(get_args())

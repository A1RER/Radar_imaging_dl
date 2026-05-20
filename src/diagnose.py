"""
diagnose.py
诊断已训练模型的内部状态：可视化每层学到的 ρ、λ、阈值，对照 MATLAB 初值。

算法展开网络的核心卖点就是"可解释"——每层对应一次 ADMM 迭代，
学到的参数直接反映网络认为哪一步需要更强/更弱的正则、阈值。

用法：
    python -m src.diagnose --ckpt checkpoints/best.pth
    python -m src.diagnose --ckpt checkpoints/best.pth --save_plot results/params.png
"""
import argparse
import math
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from .model import CADMMNet


# MATLAB 初值（对应 MMV_l1_l2_cadmmfor3.m）
MATLAB_RHO    = 400.0
MATLAB_LAM    = 150.0     # = 60000 / 400
MATLAB_THRESH = -25.0


def extract_params(model: CADMMNet):
    """从模型中提取每层的 ρ、λ、阈值，返回 numpy 数组。"""
    rhos, lams, thrs = [], [], []
    for layer in model.layers:
        rhos.append(layer.log_rho.exp().item())
        lams.append(layer.log_lam.exp().item())
        thrs.append(layer.thresh_db.item())
    return np.array(rhos), np.array(lams), np.array(thrs)


def print_table(rhos, lams, thrs):
    K = len(rhos)
    print(f'{"层":>4} {"ρ":>12} {"λ":>12} {"λ/ρ":>12} {"阈值 (dB)":>12}')
    print('-' * 60)
    for k in range(K):
        print(f'{k+1:>4} {rhos[k]:>12.2f} {lams[k]:>12.4f} '
              f'{lams[k]/rhos[k]:>12.4f} {thrs[k]:>12.2f}')
    print('-' * 60)
    print(f'{"MATLAB":>4} {MATLAB_RHO:>12.2f} {MATLAB_LAM:>12.4f} '
          f'{MATLAB_LAM/MATLAB_RHO:>12.4f} {MATLAB_THRESH:>12.2f}')


def plot_params(rhos, lams, thrs, save_path=None):
    K = len(rhos)
    xs = np.arange(1, K + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(xs, rhos, 'o-', color='#1f77b4', label='学到的 ρ')
    axes[0].axhline(MATLAB_RHO, color='gray', linestyle='--', label=f'MATLAB 初值 ({MATLAB_RHO})')
    axes[0].set_xlabel('层索引')
    axes[0].set_ylabel('ρ（惩罚系数）')
    axes[0].set_title('ρ 在各层的演化')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs, lams / rhos, 's-', color='#ff7f0e', label='学到的 λ/ρ')
    axes[1].axhline(MATLAB_LAM / MATLAB_RHO, color='gray', linestyle='--',
                    label=f'MATLAB 初值 ({MATLAB_LAM/MATLAB_RHO:.3f})')
    axes[1].set_xlabel('层索引')
    axes[1].set_ylabel('λ/ρ（软阈值）')
    axes[1].set_title('软阈值在各层的演化')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(xs, thrs, '^-', color='#2ca02c', label='学到的阈值')
    axes[2].axhline(MATLAB_THRESH, color='gray', linestyle='--',
                    label=f'MATLAB 初值 ({MATLAB_THRESH} dB)')
    axes[2].set_xlabel('层索引')
    axes[2].set_ylabel('一致性阈值 (dB)')
    axes[2].set_title('一致性阈值在各层的演化')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'参数曲线已保存：{save_path}')
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    K = ckpt['K']
    model = CADMMNet(K=K, n_apertures=3).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f'模型：{args.ckpt}  (K={K} 层)')
    if 'val_psnr' in ckpt:
        print(f'验证 PSNR：{ckpt["val_psnr"]:.2f} dB，epoch={ckpt.get("epoch","?")}')
    print()

    rhos, lams, thrs = extract_params(model)
    print_table(rhos, lams, thrs)

    if args.save_plot or args.show:
        plot_params(rhos, lams, thrs, save_path=args.save_plot)


def get_args():
    p = argparse.ArgumentParser(description='诊断深度展开 CADMM 网络的学习参数')
    p.add_argument('--ckpt',      default='checkpoints/best.pth')
    p.add_argument('--save_plot', default=None, help='保存参数曲线图路径（如 results/params.png）')
    p.add_argument('--show',      action='store_true', help='显示图像（默认仅打印表格）')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())

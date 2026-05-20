"""
train.py
训练深度展开 CADMM 网络。

用法：
    python -m src.train                                       # 默认配置
    python -m src.train --K 10 --epochs 50                    # 自定义层数和轮次
    python -m src.train --aug_snr 5 25                        # 启用在线加噪增强
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import build_loaders
from .model   import CADMMNet
from .losses  import CombinedLoss
from .utils   import psnr, seed_everything


# ── 单 epoch 训练 ─────────────────────────────────────────────────────

def train_epoch(model, loader, A, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    A = A.to(device)
    for batch_idx, (sig, z_clean) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sig     = sig.to(device)
        z_clean = z_clean.to(device)

        optimizer.zero_grad()
        z_pred = model(sig, A)
        loss   = criterion(z_pred, z_clean)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ── 验证 / 测试 ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, A, criterion, device, max_batches=None):
    model.eval()
    total_loss  = 0.0
    total_psnr  = 0.0
    n_batches   = 0
    A = A.to(device)
    for batch_idx, (sig, z_clean) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sig     = sig.to(device)
        z_clean = z_clean.to(device)
        z_pred  = model(sig, A)
        total_loss += criterion(z_pred, z_clean).item()
        total_psnr += psnr(z_pred, z_clean).item()
        n_batches += 1
    n = max(n_batches, 1)
    return total_loss / n, total_psnr / n


# ── 主训练循环 ────────────────────────────────────────────────────────

def train(args):
    # 复现性
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}，随机种子：{args.seed}')

    # 数据 + 在线增强
    aug = tuple(args.aug_snr) if args.aug_snr is not None else None
    train_loader, val_loader, test_loader, A, kx = build_loaders(
        train_mat     = args.train_mat,
        test_mat      = args.test_mat,
        batch_size    = args.batch_size,
        num_workers   = 0,
        train_augment = aug,
    )
    print(f'训练集 {len(train_loader.dataset)} 样本，'
          f'验证集 {len(val_loader.dataset)} 样本，'
          f'测试集 {len(test_loader.dataset)} 样本')
    if aug:
        print(f'已启用在线加噪增强：SNR ∈ [{aug[0]}, {aug[1]}] dB')

    # 模型 + 损失
    model = CADMMNet(K=args.K, n_apertures=3).to(device)
    print(f'模型参数量：{model.count_parameters()}（K={args.K} 层）')

    criterion = CombinedLoss(
        w_mse  = args.w_mse,
        w_l1   = args.w_l1,
        w_ssim = args.w_ssim,
    )
    print(f'损失：MSE×{args.w_mse} + L1×{args.w_l1} + (1-SSIM)×{args.w_ssim}')

    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    ckpt_path = os.path.join(args.save_dir, 'best.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss          = train_epoch(model, train_loader, A, optimizer, criterion, device,
                                          max_batches=args.max_train_batches)
        val_loss, val_psnr  = evaluate(  model, val_loader,    A,            criterion, device,
                                          max_batches=args.max_eval_batches)
        scheduler.step()

        print(f'[{epoch:3d}/{args.epochs}] '
              f'train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_PSNR={val_psnr:.2f} dB')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':      epoch,
                'K':          args.K,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   val_loss,
                'val_psnr':   val_psnr,
                'args':       vars(args),
            }, ckpt_path)
            print(f'  → 保存最优模型 (val_loss={val_loss:.4f})')

    # 最终测试集评估
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['state_dict'])
    test_loss, test_psnr = evaluate(model, test_loader, A, criterion, device,
                                    max_batches=args.max_eval_batches)
    print(f'\n测试集结果：loss={test_loss:.4f}，PSNR={test_psnr:.2f} dB')


# ── CLI 入口 ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='训练深度展开 CADMM 网络')
    # 数据
    p.add_argument('--train_mat',  default='data/train.mat')
    p.add_argument('--test_mat',   default='data/test.mat')
    p.add_argument('--aug_snr',    type=float, nargs=2, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='在线加噪 SNR 范围 (dB)，如 --aug_snr 5 25；不填则关闭')
    # 模型
    p.add_argument('--K',          type=int,   default=10,    help='展开层数')
    # 损失权重
    p.add_argument('--w_mse',      type=float, default=1.0)
    p.add_argument('--w_l1',       type=float, default=1e-4)
    p.add_argument('--w_ssim',     type=float, default=0.1)
    # 训练
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--save_dir',   default='checkpoints')
    p.add_argument('--max_train_batches', type=int, default=None,
                   help='调试/汇报用：每个 epoch 最多训练多少个 batch；默认跑完整训练集')
    p.add_argument('--max_eval_batches',  type=int, default=None,
                   help='调试/汇报用：验证/测试最多评估多少个 batch；默认跑完整数据集')
    return p.parse_args()


if __name__ == '__main__':
    train(get_args())

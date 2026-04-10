"""
train.py
训练深度展开 CADMM 网络。

用法：
    python -m src.train                      # 使用默认配置
    python -m src.train --K 10 --epochs 50   # 自定义层数和轮次
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .dataset import build_loaders
from .model   import CADMMNet
from .utils   import psnr


# ── 损失函数 ──────────────────────────────────────────────────────────

def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE + 归一化 L1（促稀疏）的加权组合，均在幅度域计算。
    pred / target: [B, nx, nkr] float
    """
    mse  = nn.functional.mse_loss(pred, target)
    l1   = torch.mean(torch.abs(pred))
    return mse + 1e-4 * l1


# ── 单 epoch 训练 ─────────────────────────────────────────────────────

def train_epoch(model, loader, A, optimizer, device):
    model.train()
    total_loss = 0.0
    A = A.to(device)
    for sig, z_clean in loader:
        sig     = sig.to(device)
        z_clean = z_clean.to(device)

        optimizer.zero_grad()
        z_pred = model(sig, A)
        loss   = loss_fn(z_pred, z_clean)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── 验证 / 测试 ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, A, device):
    model.eval()
    total_loss  = 0.0
    total_psnr  = 0.0
    A = A.to(device)
    for sig, z_clean in loader:
        sig     = sig.to(device)
        z_clean = z_clean.to(device)
        z_pred  = model(sig, A)
        total_loss += loss_fn(z_pred, z_clean).item()
        total_psnr += psnr(z_pred, z_clean).item()
    n = len(loader)
    return total_loss / n, total_psnr / n


# ── 主训练循环 ────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')

    # 数据
    train_loader, val_loader, test_loader, A, kx = build_loaders(
        train_mat    = args.train_mat,
        test_mat     = args.test_mat,
        batch_size   = args.batch_size,
        num_workers  = 0,
    )
    print(f'训练集 {len(train_loader.dataset)} 样本，'
          f'验证集 {len(val_loader.dataset)} 样本，'
          f'测试集 {len(test_loader.dataset)} 样本')

    # 模型
    model = CADMMNet(K=args.K, n_apertures=3).to(device)
    print(f'模型参数量：{model.count_parameters()}（K={args.K} 层）')

    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, A, optimizer, device)
        val_loss, val_psnr = evaluate(model, val_loader, A, device)
        scheduler.step()

        print(f'[{epoch:3d}/{args.epochs}] '
              f'train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_PSNR={val_psnr:.2f} dB')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, 'best.pth')
            torch.save({
                'epoch':      epoch,
                'K':          args.K,
                'state_dict': model.state_dict(),
                'val_loss':   val_loss,
                'val_psnr':   val_psnr,
            }, ckpt_path)
            print(f'  → 已保存最优模型 (val_loss={val_loss:.4f})')

    # 最终测试集评估
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
    test_loss, test_psnr = evaluate(model, test_loader, A, device)
    print(f'\n测试集结果：loss={test_loss:.4f}，PSNR={test_psnr:.2f} dB')


# ── CLI 入口 ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='训练深度展开 CADMM 网络')
    p.add_argument('--train_mat',  default='data/train.mat')
    p.add_argument('--test_mat',   default='data/test.mat')
    p.add_argument('--K',          type=int,   default=10,    help='展开层数')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--save_dir',   default='checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    train(get_args())

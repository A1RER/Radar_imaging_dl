# Radar Imaging with Deep Unrolling

[English](#english) | [中文](#中文)

---

## English

Deep learning-based radar imaging via algorithm unrolling of Consensus-ADMM.  
Targets OFDM-waveform ISAR/SAR scenarios with multiple moving vehicles.

### Overview

Traditional compressed-sensing imaging (Consensus-ADMM) requires hand-tuned hyperparameters and a fixed number of iterations. This project unrolls each ADMM iteration into a learnable network layer, allowing end-to-end training to replace manual tuning while significantly reducing inference iterations.

```
Input: 3 sub-aperture compressed range profiles  [3, nkr, nkx]
         ↓  K unrolled ADMM layers (K ≪ 30)
Output: reconstructed radar image                [nx,  nkr]
```

Each layer learns its own regularization strength λ, penalty coefficient ρ, and consensus threshold — all initialized from the classical ADMM values.

### Project Structure

```
Radar_imaging_dl/
├── matlab/
│   └── generate_data.m     # Simulate OFDM radar returns and export training data
├── src/
│   ├── model.py            # Deep unrolled CADMM network
│   ├── dataset.py          # PyTorch Dataset for .mat files
│   ├── train.py            # Training loop with CLI
│   ├── evaluate.py         # Evaluation vs. traditional CADMM + visualization
│   └── utils.py            # dB conversion, soft-threshold, PSNR, imaging entropy
├── data/                   # Generated .mat files (not tracked)
├── checkpoints/            # Saved model weights (not tracked)
├── results/                # Output comparison figures (not tracked)
└── requirements.txt
```

### Requirements

```
Python >= 3.10
```

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| torch | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |
| tqdm | ≥ 4.65 |

### Usage

#### 1. Generate Training Data

Run `matlab/generate_data.m` in MATLAB (set working directory to `matlab/`).  
This simulates 3-vehicle OFDM radar returns at multiple SNR levels and saves:

```
data/train.mat   — training set
data/test.mat    — test set
```

Each `.mat` contains:
- `tr_sig / te_sig`  `[N, 3, nkr, nkx]` complex — sub-aperture inputs
- `tr_z   / te_z`    `[N, nx,  nkr]`    float   — clean reference images (labels)
- `A`                `[nkx, nx]`        complex — sensing matrix
- `kx`               `[nkx]`            float   — wavenumber vector

#### 2. Train

```bash
python -m src.train
```

Key options:

```
--K           int    Number of unrolled layers   (default: 10)
--epochs      int    Training epochs             (default: 100)
--batch_size  int    Batch size                  (default: 8)
--lr          float  Learning rate               (default: 1e-3)
--save_dir    str    Checkpoint directory        (default: checkpoints/)
```

Best checkpoint is saved to `checkpoints/best.pth`.

#### 3. Evaluate

```bash
python -m src.evaluate
```

Outputs a table comparing the proposed network against traditional CADMM on PSNR and imaging entropy, and saves side-by-side comparison figures to `results/`.

### Method

The network unrolls the following Consensus-ADMM update equations into K trainable layers:

$$x_i^{(k)} = (A^H A + \rho_k I)^{-1}(A^H y_i - o_i^{(k)} + \rho_k z^{(k)})$$

$$z^{(k+1)} = \mathcal{S}_{\lambda_k/\rho_k}\!\left(\text{mask}_k \cdot \frac{1}{M}\sum_i |x_i^{(k)} + o_i^{(k)}/\rho_k|\right)$$

$$o_i^{(k+1)} = o_i^{(k)} + \rho_k(x_i^{(k)} - z^{(k+1)})$$

where $\mathcal{S}$ is soft-thresholding, and the consensus mask is replaced by a differentiable sigmoid approximation to allow gradient flow.

Learnable parameters per layer: `log_rho`, `log_lam`, `thresh_db` — initialized from classical ADMM values (ρ=400, λ=60000, threshold=−25 dB).

### Metrics

| Metric | Description |
|---|---|
| PSNR (dB) ↑ | Peak signal-to-noise ratio in amplitude domain |
| Imaging Entropy ↓ | Lower entropy = better focused image |

### License

MIT © 2026 Leslie Shen

---

## 中文

基于共识-ADMM 算法展开的深度学习雷达成像。  
面向 OFDM 波形 ISAR/SAR 场景，支持多运动目标。

### 概述

传统压缩感知成像（共识-ADMM）需要手动调参，且迭代次数固定。本项目将每次 ADMM 迭代展开为一个可学习的网络层，通过端到端训练取代人工调参，同时大幅减少推理所需的迭代次数。

```
输入：3 个子孔径压缩距离像  [3, nkr, nkx]
         ↓  K 层展开 ADMM（K ≪ 30）
输出：重建雷达图像            [nx,  nkr]
```

每一层独立学习正则化强度 λ、惩罚系数 ρ 以及共识阈值，均以经典 ADMM 值为初始化。

### 项目结构

```
Radar_imaging_dl/
├── matlab/
│   └── generate_data.m     # 仿真 OFDM 雷达回波并导出训练数据
├── src/
│   ├── model.py            # 深度展开 CADMM 网络
│   ├── dataset.py          # 读取 .mat 文件的 PyTorch Dataset
│   ├── train.py            # 训练循环（含命令行接口）
│   ├── evaluate.py         # 与传统 CADMM 对比评估及可视化
│   └── utils.py            # dB 转换、软阈值、PSNR、成像熵
├── data/                   # 生成的 .mat 文件（不入库）
├── checkpoints/            # 模型权重（不入库）
├── results/                # 对比图输出（不入库）
└── requirements.txt
```

### 环境要求

```
Python >= 3.10
```

```bash
pip install -r requirements.txt
```

| 依赖包 | 版本要求 |
|---|---|
| torch | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |
| tqdm | ≥ 4.65 |

### 使用方法

#### 1. 生成训练数据

在 MATLAB 中运行 `matlab/generate_data.m`（将工作目录切换至 `matlab/`）。  
该脚本仿真三辆运动车辆在多 SNR 条件下的 OFDM 雷达回波，并保存：

```
data/train.mat   — 训练集
data/test.mat    — 测试集
```

每个 `.mat` 文件包含：
- `tr_sig / te_sig`  `[N, 3, nkr, nkx]` 复数 — 子孔径输入
- `tr_z   / te_z`    `[N, nx,  nkr]`    浮点 — 无噪声参考图像（标签）
- `A`                `[nkx, nx]`        复数 — 感知矩阵
- `kx`               `[nkx]`            浮点 — 波数向量

#### 2. 训练

```bash
python -m src.train
```

主要参数：

```
--K           int    展开层数           （默认：10）
--epochs      int    训练轮次           （默认：100）
--batch_size  int    批大小             （默认：8）
--lr          float  学习率             （默认：1e-3）
--save_dir    str    权重保存目录       （默认：checkpoints/）
```

最优模型权重保存至 `checkpoints/best.pth`。

#### 3. 评估

```bash
python -m src.evaluate
```

输出本网络与传统 CADMM 在 PSNR 和成像熵上的对比表格，并将并排对比图保存至 `results/`。

### 方法

网络将以下共识-ADMM 迭代公式展开为 K 个可训练层：

$$x_i^{(k)} = (A^H A + \rho_k I)^{-1}(A^H y_i - o_i^{(k)} + \rho_k z^{(k)})$$

$$z^{(k+1)} = \mathcal{S}_{\lambda_k/\rho_k}\!\left(\text{mask}_k \cdot \frac{1}{M}\sum_i |x_i^{(k)} + o_i^{(k)}/\rho_k|\right)$$

$$o_i^{(k+1)} = o_i^{(k)} + \rho_k(x_i^{(k)} - z^{(k+1)})$$

其中 $\mathcal{S}$ 为软阈值函数，共识掩模替换为可微的 sigmoid 近似以保证梯度流通。

每层可学习参数：`log_rho`、`log_lam`、`thresh_db`，初始化自经典 ADMM 值（ρ=400，λ=60000，阈值=−25 dB）。

### 评价指标

| 指标 | 说明 |
|---|---|
| PSNR (dB) ↑ | 幅度域峰值信噪比，越高越好 |
| 成像熵 ↓ | 熵越低表示图像聚焦越好 |

### 许可证

MIT © 2026 Leslie Shen

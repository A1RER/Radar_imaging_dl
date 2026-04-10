# Radar Imaging with Deep Unrolling

Deep learning-based radar imaging via algorithm unrolling of Consensus-ADMM.  
Targets OFDM-waveform ISAR/SAR scenarios with multiple moving vehicles.

## Overview

Traditional compressed-sensing imaging (Consensus-ADMM) requires hand-tuned hyperparameters and a fixed number of iterations. This project unrolls each ADMM iteration into a learnable network layer, allowing end-to-end training to replace manual tuning while significantly reducing inference iterations.

```
Input: 3 sub-aperture compressed range profiles  [3, nkr, nkx]
         ↓  K unrolled ADMM layers (K ≪ 30)
Output: reconstructed radar image                [nx,  nkr]
```

Each layer learns its own regularization strength λ, penalty coefficient ρ, and consensus threshold — all initialized from the classical ADMM values.

## Project Structure

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

## Requirements

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

## Usage

### 1. Generate Training Data

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

### 2. Train

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

### 3. Evaluate

```bash
python -m src.evaluate
```

Outputs a table comparing the proposed network against traditional CADMM on PSNR and imaging entropy, and saves side-by-side comparison figures to `results/`.

## Method

The network unrolls the following Consensus-ADMM update equations into K trainable layers:

$$x_i^{(k)} = (A^H A + \rho_k I)^{-1}(A^H y_i - o_i^{(k)} + \rho_k z^{(k)})$$

$$z^{(k+1)} = \mathcal{S}_{\lambda_k/\rho_k}\!\left(\text{mask}_k \cdot \frac{1}{M}\sum_i |x_i^{(k)} + o_i^{(k)}/\rho_k|\right)$$

$$o_i^{(k+1)} = o_i^{(k)} + \rho_k(x_i^{(k)} - z^{(k+1)})$$

where $\mathcal{S}$ is soft-thresholding, and the consensus mask is replaced by a differentiable sigmoid approximation to allow gradient flow.

Learnable parameters per layer: `log_rho`, `log_lam`, `thresh_db` — initialized from classical ADMM values (ρ=400, λ=60000, threshold=−25 dB).

## Metrics

| Metric | Description |
|---|---|
| PSNR (dB) ↑ | Peak signal-to-noise ratio in amplitude domain |
| Imaging Entropy ↓ | Lower entropy = better focused image |

## License

MIT © 2026 Leslie Shen

% generate_data.m
% 从 OFDM_ISAR 仿真生成深度学习训练/测试数据
%
% 使用方法：
%   1. 在 MATLAB 中将工作目录切换到本文件所在的 matlab/ 文件夹
%   2. 直接运行本脚本
%   3. 输出文件保存在 ../data/train.mat 和 ../data/test.mat
%
% 输出 .mat 文件变量说明：
%   tr_sig / te_sig  [N, 3, nkr, nkx] complex  - 3个子孔径距离压缩数据（网络输入）
%   tr_z   / te_z    [N, nx, nkr]     real     - 无噪声参考图像（训练标签）
%   A                [nkx, nx]        complex  - 感知矩阵（所有样本共享）
%   kx               [1, nkx]         real     - 方位波数向量

clear; close all;
addpath('../datafromlead/室外动目标成像跟踪源码-3车辆/')

%% ── 雷达系统参数（与 OFDM_ISAR.m 保持一致） ────────────────────────
c      = 3e8;
bw     = 0.8e9;
fc     = 28e9;
fs     = 0.8e9;
prf    = 4000;
xa     = 0;  ya = 400;  za = 100;
rRef   = sqrt(xa^2 + ya^2 + za^2);
NoOFDM = 512;
Ts     = 4;                                     % 总观测时间 (s)
ts     = linspace(0, Ts, Ts*prf);
nR     = NoOFDM + 1;
v      = 20;                                    % 平台速度 (m/s)

tt_sub = 0.4;                                   % 子孔径时间 (s)
per    = 0.75;                                  % 孔径重复率
nf     = (Ts/tt_sub - 1)/(1-per) + 1;          % 总帧数
nt     = round(prf*Ts / ((nf-1)*(1-per) + 1)); % 每帧慢时间采样点数
ttt    = linspace(0, tt_sub, nt);               % 每帧时间轴
rMax   = 60;                                    % 成像距离范围 (m)

%% ── 加载仿真信号 ────────────────────────────────────────────────────
load('../datafromlead/室外动目标成像跟踪源码-3车辆/sig_1_car_4s.mat');
load('../datafromlead/室外动目标成像跟踪源码-3车辆/sig_2_Trunk_4s.mat');
load('../datafromlead/室外动目标成像跟踪源码-3车辆/sig_3_car_4s.mat');
sig_clean = sig_1 + sig_2 + sig_3;

%% ── 公共预处理 ──────────────────────────────────────────────────────
fr1      = (-nR/2 : nR/2-1).' / nR * fs;
KR       = 4*(fc + fr1)/c * pi;                % 波数向量 [nR, 1]
ft_clean = fftshift(fft(sig_clean, [], 1), 1); % 距离向 FFT
Rref     = sqrt((xa - v*ttt).^2 + ya^2 + za^2);
Href_new = exp(1j * KR * (Rref - rRef));       % 参考补偿相位 [nR, nt]

%% ── 推算感知矩阵尺寸（取第一帧无噪声） ─────────────────────────────
sig_f0     = ft_clean(:, 1:nt) .* Href_new;
[skc0, kx] = interp_near_sar1(sig_f0, ttt, KR', rMax, xa, ya, v, za);
ntnew      = size(skc0, 2);
x_grid     = linspace(-rMax, rMax, ntnew);
A          = exp(-1j * kx' * x_grid * pi / rMax);  % [nkx, nx]
[nkx, nx]  = size(A);
r0         = fftshift(fft(skc0, [], 1), 1)';        % [nkx, nkr]
nkr        = size(r0, 2);
fprintf('感知矩阵 A = [%d x %d]，图像尺寸 = [%d x %d]\n', nkx, nx, nx, nkr);

%% ── SNR 配置 ────────────────────────────────────────────────────────
snr_train_db = [5, 10, 15, 20];
snr_test_db  = [8, 12, 18];

N_train = nf * length(snr_train_db);
N_test  = nf * length(snr_test_db);
tr_sig  = zeros(N_train, 3, nkr, nkx, 'like', 1+1j);
tr_z    = zeros(N_train, nx, nkr);
te_sig  = zeros(N_test,  3, nkr, nkx, 'like', 1+1j);
te_z    = zeros(N_test,  nx, nkr);

%% ── 逐帧生成样本 ────────────────────────────────────────────────────
tr_idx = 0;  te_idx = 0;
fprintf('开始生成，共 %d 帧...\n', nf);

for i_frame = 1:nf
    col_s = round(1  + nt*(1-per)*(i_frame-1));
    col_e = round(nt + nt*(1-per)*(i_frame-1));
    sig_frame = ft_clean(:, col_s:col_e);

    % 无噪声参考图像：高迭代 ADMM 输出作为训练标签
    sig_ref_c  = sig_frame .* Href_new;
    [skc_c, ~] = interp_near_sar1(sig_ref_c, ttt, KR', rMax, xa, ya, v, za);
    r_c        = fftshift(fft(skc_c, [], 1), 1)';
    z_ref      = run_admm(A, r_c, 80, 60000, 400);  % [nx, nkr]

    for snr_db = snr_train_db
        tr_idx = tr_idx + 1;
        tr_sig(tr_idx,:,:,:) = make_sample(sig_frame, Href_new, ttt, KR, rMax, xa, ya, v, za, snr_db);
        tr_z(tr_idx,:,:)     = z_ref;
    end

    for snr_db = snr_test_db
        te_idx = te_idx + 1;
        te_sig(te_idx,:,:,:) = make_sample(sig_frame, Href_new, ttt, KR, rMax, xa, ya, v, za, snr_db);
        te_z(te_idx,:,:)     = z_ref;
    end

    if mod(i_frame, 5) == 0
        fprintf('  %d / %d 帧\n', i_frame, nf);
    end
end

%% ── 保存 ────────────────────────────────────────────────────────────
if ~exist('../data', 'dir'), mkdir('../data'); end
save('../data/train.mat', 'tr_sig', 'tr_z', 'A', 'kx', '-v7.3');
save('../data/test.mat',  'te_sig', 'te_z', 'A', 'kx', '-v7.3');
fprintf('完成！训练集 %d 样本，测试集 %d 样本\n', N_train, N_test);


%% ════════════════════════════════════════════════════════════════════
%% 辅助函数
%% ════════════════════════════════════════════════════════════════════

function sample = make_sample(sig_frame, Href_new, ttt, KR, rMax, xa, ya, v, za, snr_db)
% 对单帧信号独立加噪三次，模拟 3 个子孔径快拍，返回 [3, nkr, nkx]
    for k = 1:3
        sig_nk     = add_noise(sig_frame, snr_db);
        sig_rk     = sig_nk .* Href_new;
        [skc, ~]   = interp_near_sar1(sig_rk, ttt, KR', rMax, xa, ya, v, za);
        rk         = fftshift(fft(skc, [], 1), 1)';   % [nkx, nkr]
        if k == 1
            [nkx, nkr] = size(rk);
            sample = zeros(3, nkr, nkx, 'like', 1+1j);
        end
        sample(k,:,:) = rk.';   % [nkr, nkx]
    end
end

function z = run_admm(A, y, max_iter, lamda, rho)
% Consensus-ADMM（单目标），用于生成无噪声参考图像
% A: [nkx, nx],  y: [nkx, nkr]
    [~, nx] = size(A);
    nkr     = size(y, 2);
    u       = lamda / rho;
    H_inv   = inv(A'*A + rho*eye(nx));
    Aty     = A' * y;
    o       = zeros(nx, nkr);
    z       = zeros(nx, nkr);
    for k = 1:max_iter
        xt  = H_inv * (Aty - o + rho*z);
        a   = xt + o/rho;
        a_n = 20*log10(abs(a) ./ (max(abs(a(:)))+1e-10) + 1e-10);
        idx = double(a_n > -25);
        z   = max(idx .* abs(a) - u, 0);
        o   = o + rho*(xt - z);
    end
end

function out = add_noise(sig, snr_db)
% 向复数信号添加加性高斯白噪声
    p_sig   = mean(abs(sig(:)).^2);
    p_noise = p_sig / (10^(snr_db/10));
    noise   = sqrt(p_noise/2) * (randn(size(sig)) + 1j*randn(size(sig)));
    out     = sig + noise;
end

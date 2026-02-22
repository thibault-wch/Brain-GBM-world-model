import os
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm
from math import exp


# =========================
#   SSIM3D 实现（3D卷积）
# =========================

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # [W,1]
    _2D_window = _1D_window.mm(_1D_window.t())            # [W,W]
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)) \
        .reshape(window_size, window_size, window_size) \
        .float().unsqueeze(0).unsqueeze(0)                # [1,1,W,W,W]
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim_3D(img1, img2, window, data_range, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def ssim3D(img1, img2, data_range, window_size=11, size_average=True):
    """
    img1,img2: [N, C, D, H, W]
    """
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, data_range, window_size, channel, size_average)


# =========================
#   3D 误差指标（体素级）
# =========================

def cal_mae(x, y):
    return torch.abs(x - y).mean().item()


def cal_mse(x, y):
    return ((x - y) ** 2).mean().item()


def cal_rmse(x, y):
    return (cal_mse(x, y) ** 0.5)


def cal_nmse(x, y, eps=1e-8):
    """
    NMSE = ||x - y||^2 / ||x||^2
    x: ground truth
    """
    num = ((x - y) ** 2).sum()
    den = (x ** 2).sum().clamp_min(eps)
    return (num / den).item()


def cal_psnr_from_mse(mse, data_range=2.0, eps=1e-12):
    mse = max(float(mse), eps)
    return 10.0 * torch.log10(torch.tensor((data_range ** 2) / mse)).item()


def cal_metrics_3d(gt, pred, data_range=2.0, window_size=11):
    """
    gt, pred: [N, 1, D, H, W] in [-1,1]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = gt.to(device)
    pred = pred.to(device)

    mae = cal_mae(gt, pred)
    mse = ((gt - pred) ** 2).mean()         # tensor
    mse_val = mse.item()
    rmse = (mse.sqrt()).item()
    psnr = cal_psnr_from_mse(mse_val, data_range=data_range)
    nmse = cal_nmse(gt, pred)

    ssim_val = ssim3D(gt, pred, data_range=data_range, window_size=window_size).item()

    return mae, psnr, ssim_val, nmse, rmse


# =========================
#   数据读取
# =========================

def get_data_pt(datapath):
    """
    读取 .pt，并确保 shape 为 [1, 3, D, H, W] 或 [3, D, H, W]
    """
    data = torch.load(datapath)
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{datapath} 中不是 Tensor")

    if data.dim() == 4:
        data = data.unsqueeze(0)  # [C,D,H,W] -> [1,C,D,H,W]
    elif data.dim() == 5:
        pass
    else:
        raise ValueError(f"{datapath} shape={data.shape}，期望 [1,3,D,H,W] 或 [3,D,H,W]")

    return data.float()


# =========================
#   主评估逻辑
# =========================

def evaluate_pt_folders(real_image_path,
                        generated_image_path,
                        out_results_dir,
                        methodname,
                        data_range=2.0,
                        window_size=11):

    real_files = [f for f in os.listdir(real_image_path) if f.endswith('.pt')]
    gen_files = [f for f in os.listdir(generated_image_path) if f.endswith('.pt')]
    common_files = sorted(set(real_files) & set(gen_files))

    if len(common_files) == 0:
        raise RuntimeError("两个文件夹下没有同名 .pt 文件")

    modality_names = ['flair', 't1ce', 't2w']
    results = []

    for fname in tqdm(common_files, desc="Evaluating .pt files"):
        real_data = get_data_pt(os.path.join(real_image_path, fname))
        gen_data = get_data_pt(os.path.join(generated_image_path, fname))

        if real_data.shape != gen_data.shape:
            print(f"Warning: {fname} shape mismatch: real={real_data.shape}, gen={gen_data.shape}, skip")
            continue
        if real_data.shape[1] != 3:
            print(f"Warning: {fname} channel != 3 (got {real_data.shape[1]}), skip")
            continue

        case_name = os.path.splitext(fname)[0]

        for c in range(3):
            modality = modality_names[c]
            gt = real_data[:, c:c+1, ...]   # [N,1,D,H,W]
            pred = gen_data[:, c:c+1, ...]

            mae, psnr, ssim_val, nmse, rmse = cal_metrics_3d(
                gt, pred, data_range=data_range, window_size=window_size
            )

            results.append({
                'case': case_name,
                'filename': fname,
                'modality': modality,
                'channel_idx': c,
                'mae': mae,
                'psnr': psnr,
                'ssim3d': ssim_val,
                'nmse': nmse,
                'rmse': rmse,
            })

    os.makedirs(out_results_dir, exist_ok=True)
    df = pd.DataFrame(results)

    # 1) per-item（长表）
    per_item_path = os.path.join(out_results_dir, f"{methodname}_per_item.csv")
    df.to_csv(per_item_path, index=False)
    print(f"Saved per-item table: {per_item_path}")

    # 2) 一个总表：按 modality 统计 mean/var + overall
    metrics = ['mae', 'psnr', 'ssim3d', 'nmse', 'rmse']
    by_mod = df.groupby('modality')[metrics].agg(['mean', 'var'])
    by_mod.columns = [f"{m}_{s}" for m, s in by_mod.columns]
    by_mod = by_mod.reset_index()

    overall_mean = df[metrics].mean()
    overall_var = df[metrics].var()

    overall_row = {'modality': 'overall'}
    for m in metrics:
        overall_row[f"{m}_mean"] = float(overall_mean[m])
        overall_row[f"{m}_var"] = float(overall_var[m])

    summary = pd.concat([by_mod, pd.DataFrame([overall_row])], ignore_index=True)

    summary_path = os.path.join(out_results_dir, f"{methodname}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")


# =========================
#   CLI
# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_results_dir', type=str, default='./logs')
    parser.add_argument('-methodname', type=str, default='all_new')
    parser.add_argument('-data_range', type=float, default=2.0,
                        help='PSNR data range. For [-1,1], use 2.0')
    parser.add_argument('-window_size', type=int, default=11,
                        help='SSIM3D window size')
    parser.add_argument('-real_image_path', type=str, default='/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/test/wo_seg_new/recons_xt/')
    parser.add_argument('-generated_image_path', type=str, default='/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/test/all_new/predicted/')
    args = parser.parse_args()

    evaluate_pt_folders(
        args.real_image_path,
        args.generated_image_path,
        args.out_results_dir,
        args.methodname,
        data_range=args.data_range,
        window_size=args.window_size
    )

import argparse
import csv
import os
import random
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from snn_tracker import SpikingConvLIFTracker


ORIG_W, ORIG_H = 346, 260
GRID_W, GRID_H = 86, 65
DT_STEP = 0.005  # 5 ms bins (matches Kalman baseline)


def load_pass_tensors(h5_path: str, gt_path: str):
    """Bin raw events into (T, 2, GRID_H, GRID_W) and align ground-truth (T, 2)."""
    with h5py.File(h5_path, "r") as f:
        ev = f["events"][:]

    t_ev = ev[:, 0].astype(np.float64) * 1e-6
    x_ev = ev[:, 1].astype(np.float64)
    y_ev = ev[:, 2].astype(np.float64)
    p_ev = ev[:, 3].astype(np.int64)

    gt_t, gt_x, gt_y = [], [], []
    with open(gt_path, "r") as f:
        for row in csv.DictReader(f):
            gt_t.append(float(row["timestamp_s"]))
            gt_x.append(float(row["x"]))
            gt_y.append(float(row["y"]))
    gt_t, gt_x, gt_y = np.array(gt_t), np.array(gt_x), np.array(gt_y)

    t0, t1 = float(gt_t[0]), float(gt_t[-1])
    n_bins = int(round((t1 - t0) / DT_STEP))
    frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)

    bin_idx = np.floor((t_ev - t0) / DT_STEP).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    b = bin_idx[valid]
    gx = np.clip((x_ev[valid] * (GRID_W / ORIG_W)).astype(np.int64), 0, GRID_W - 1)
    gy = np.clip((y_ev[valid] * (GRID_H / ORIG_H)).astype(np.int64), 0, GRID_H - 1)
    ch = (p_ev[valid] > 0).astype(np.int64)  # 1 = ON, 0 = OFF

    np.add.at(frames, (b, ch, gy, gx), 1.0)
    frames = np.log1p(frames)

    t_centers = t0 + (np.arange(n_bins) + 0.5) * DT_STEP
    ref_x = np.interp(t_centers, gt_t, gt_x)
    ref_y = np.interp(t_centers, gt_t, gt_y)
    targets = np.stack([ref_x, ref_y], axis=-1).astype(np.float32)

    return torch.from_numpy(frames), torch.from_numpy(targets)


def augment_pass(x: torch.Tensor, y: torch.Tensor):
    """Apply random horizontal and vertical flips to (1, T, 2, H, W) and (1, T, 2)."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
        y = y.clone()
        y[..., 0] = (ORIG_W - 1.0) - y[..., 0]
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-2])
        y = y.clone()
        y[..., 1] = (ORIG_H - 1.0) - y[..., 1]
    return x, y


def gaussian_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    sigma_px: float = 6.0,
):
    """Cross-entropy between predicted spatial distribution and a sharp 2D Gaussian target."""
    dx2 = (grid_x.unsqueeze(1) - targets[..., 0:1]) ** 2
    dy2 = (grid_y.unsqueeze(1) - targets[..., 1:2]) ** 2
    target_dist = torch.exp(-(dx2 + dy2) / (2.0 * sigma_px**2))
    target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def evaluate_split(model, recs, device):
    model.eval()
    rmses, spreads = [], []
    with torch.no_grad():
        for rec in recs:
            x = rec["frames"].unsqueeze(0).to(device)
            y = rec["targets"].unsqueeze(0).to(device)
            pred, _, spread = model(x, return_spread=True)
            err = torch.hypot(pred[..., 0] - y[..., 0], pred[..., 1] - y[..., 1])
            rmses.append(torch.sqrt((err**2).mean()).item())
            spreads.append(spread.mean().item())
    return float(np.mean(rmses)), float(np.mean(spreads))


def main():
    parser = argparse.ArgumentParser(description="Train SpikingConvLIFTracker v3.")
    parser.add_argument("--manifest", default="data/benchmark_v2/manifest.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--hm_weight", type=float, default=2.0)
    parser.add_argument("--sigma_px", type=float, default=6.0)
    parser.add_argument("--out_ckpt", default="checkpoints/snn_tracker_v3_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    records = []
    with open(args.manifest, "r") as f:
        for row in csv.DictReader(f):
            frames, targets = load_pass_tensors(row["events_h5"], row["ground_truth_csv"])
            records.append({**row, "frames": frames, "targets": targets})

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=args.beta,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    coord_criterion = nn.SmoothL1Loss(beta=5.0)

    best_val_rmse = float("inf")
    print(
        f"Training SpikingConvLIFTracker v3 (beta={args.beta}, hm_w={args.hm_weight}) on {device} | "
        f"Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}"
    )
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_recs)
        train_loss = 0.0

        for rec in train_recs:
            x = rec["frames"].unsqueeze(0).to(device)
            y = rec["targets"].unsqueeze(0).to(device)
            x, y = augment_pass(x, y)

            optimizer.zero_grad()
            pred, _, logits = model(x, return_logits=True)
            c_loss = coord_criterion(pred, y)
            h_loss = gaussian_heatmap_loss(logits, y, model.grid_x, model.grid_y, sigma_px=args.sigma_px)
            loss = c_loss + args.hm_weight * h_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        val_rmse, val_spread = evaluate_split(model, val_recs, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), args.out_ckpt)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss / len(train_recs):6.2f} | "
                f"Val RMSE: {val_rmse:6.2f} px | Val Spread: {val_spread:5.1f} px | Best Val: {best_val_rmse:6.2f} px"
            )

    print(f"\nTraining finished in {time.time() - t_start:.1f}s | Best Val RMSE: {best_val_rmse:.2f} px")
    model.load_state_dict(torch.load(args.out_ckpt, map_location=device, weights_only=True))
    model.eval()

    by_regime = {}
    print(f"\n=== Unseen Test Split ({len(test_recs)} passes) ===")
    print(f"{'Pass ID':<30} | {'Regime':<15} | {'Speed':>7} | {'RMSE (px)':>9} | {'Spread':>7} | {'Spike %':>7}")
    print("-" * 90)

    with torch.no_grad():
        for rec in test_recs:
            x = rec["frames"].unsqueeze(0).to(device)
            y = rec["targets"].unsqueeze(0).to(device)
            pred, spk_rate, spread = model(x, return_spread=True)
            err = torch.hypot(pred[..., 0] - y[..., 0], pred[..., 1] - y[..., 1])
            rmse = torch.sqrt((err**2).mean()).item()
            sp_mean = spread.mean().item()
            spk_pct = 100.0 * spk_rate.item()
            by_regime.setdefault(rec["regime"], []).append((rmse, sp_mean, spk_pct))
            print(
                f"{rec['pass_id']:<30} | {rec['regime']:<15} | {float(rec['speed_px_s']):>7.1f} | "
                f"{rmse:>9.2f} | {sp_mean:>6.1f}px | {spk_pct:>6.2f}%"
            )

    print("\n=== SNN v3 Test Regime Summary ===")
    for reg, vals in by_regime.items():
        print(
            f"{reg:<16} | Mean RMSE: {np.mean([v[0] for v in vals]):6.2f} px | "
            f"Mean Spread: {np.mean([v[1] for v in vals]):5.1f} px | Mean Spike Rate: {np.mean([v[2] for v in vals]):5.2f}%"
        )


if __name__ == "__main__":
    main()

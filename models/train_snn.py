import csv
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = "data/benchmark_v1/manifest.csv"

    records = []
    with open(manifest_path, "r") as f:
        for row in csv.DictReader(f):
            frames, targets = load_pass_tensors(row["events_h5"], row["ground_truth_csv"])
            records.append({**row, "frames": frames, "targets": targets})

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]

    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=0.85,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=5.0)
    epochs = 60

    print(f"Training SpikingConvLIFTracker on {device} | Train: {len(train_recs)}, Val: {len(val_recs)}")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for rec in train_recs:
            x = rec["frames"].unsqueeze(0).to(device)
            y = rec["targets"].unsqueeze(0).to(device)

            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_rmses = []
            with torch.no_grad():
                for rec in val_recs:
                    x = rec["frames"].unsqueeze(0).to(device)
                    y = rec["targets"].unsqueeze(0).to(device)
                    pred, _ = model(x)
                    err = torch.hypot(pred[..., 0] - y[..., 0], pred[..., 1] - y[..., 1])
                    val_rmses.append(torch.sqrt((err**2).mean()).item())
            print(
                f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss / len(train_recs):6.2f} | "
                f"Val RMSE: {np.mean(val_rmses):6.2f} px"
            )

    print(f"\nTraining finished in {time.time() - t_start:.1f}s")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/snn_tracker_v1.pt")

    model.eval()
    by_regime = {}
    print(f"\n{'Pass ID':<30} | {'Split':<5} | {'Regime':<15} | {'RMSE (px)':>9} | {'Spike %':>7}")
    print("-" * 80)

    with torch.no_grad():
        for rec in records:
            x = rec["frames"].unsqueeze(0).to(device)
            y = rec["targets"].unsqueeze(0).to(device)
            pred, spk_rate = model(x)
            err = torch.hypot(pred[..., 0] - y[..., 0], pred[..., 1] - y[..., 1])
            rmse = torch.sqrt((err**2).mean()).item()
            spk_pct = 100.0 * spk_rate.item()
            by_regime.setdefault(rec["regime"], []).append((rmse, spk_pct))
            print(
                f"{rec['pass_id']:<30} | {rec['split']:<5} | {rec['regime']:<15} | "
                f"{rmse:>9.2f} | {spk_pct:>6.2f}%"
            )

    print("\n=== SNN Regime Summary ===")
    for reg, vals in by_regime.items():
        rmses = [v[0] for v in vals]
        spks = [v[1] for v in vals]
        print(f"{reg:<16} | Mean RMSE: {np.mean(rmses):6.2f} px | Mean Spike Rate: {np.mean(spks):5.2f}%")


if __name__ == "__main__":
    main()

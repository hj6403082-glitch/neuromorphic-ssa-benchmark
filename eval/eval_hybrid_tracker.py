import csv
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("baselines"))
sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))

from kalman_tracker import ConstantVelocityKalman2D
from eval_kalman_benchmark import evaluate_pass as eval_kalman_sim
from train_snn import load_pass_tensors, ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP
from snn_tracker import SpikingConvLIFTracker
from eval_ebssa_zeroshot import load_ebssa_recording, run_kalman as eval_kalman_ebssa


def run_hybrid_from_snn_preds(
    est_t: np.ndarray,
    snn_x: np.ndarray,
    snn_y: np.ndarray,
    snn_spread: np.ndarray,
    max_spread_px: float = 35.0,
    gate_radius: float = 30.0,
    reacq_steps: int = 12,
):
    """Uncertainty-gated Constant-Velocity Kalman filter driven by SNN (x, y) + spread."""
    n = len(est_t)
    hyb_x = np.zeros(n, dtype=np.float64)
    hyb_y = np.zeros(n, dtype=np.float64)

    kf = None
    updates = 0
    missed = 0

    for i in range(n):
        t = float(est_t[i])
        sx, sy, sp = float(snn_x[i]), float(snn_y[i]), float(snn_spread[i])
        confident = sp <= max_spread_px

        if kf is None:
            hyb_x[i], hyb_y[i] = sx, sy
            # Initialize with zero velocity so 5 ms jitter never launches the filter off-screen
            if confident:
                kf = ConstantVelocityKalman2D(sx, sy, t, vx0=0.0, vy0=0.0)
                updates += 1
                missed = 0
        else:
            kf.predict(t)
            updated = False
            if confident:
                updated = kf.update(sx, sy, gate_radius=gate_radius)

            if updated:
                updates += 1
                missed = 0
            else:
                missed += 1
                # Re-acquire cleanly if coasting too long and SNN has a sharp lock
                if missed >= reacq_steps and confident:
                    kf = ConstantVelocityKalman2D(sx, sy, t, vx0=0.0, vy0=0.0)
                    updates += 1
                    missed = 0

            hyb_x[i], hyb_y[i] = kf.state[0], kf.state[1]

    upd_pct = 100.0 * updates / max(1, n)
    return hyb_x, hyb_y, upd_pct


def infer_snn_with_spread(model, frames: np.ndarray, device: torch.device):
    with torch.no_grad():
        x_in = torch.from_numpy(frames).unsqueeze(0).to(device)
        pred, spk_rate, spread = model(x_in, return_spread=True)
    return (
        pred.squeeze(0).cpu().numpy(),
        spread.squeeze(0).cpu().numpy(),
        100.0 * spk_rate.item(),
    )


def benchmark_synthetic(manifest_path: str, model, device: torch.device):
    by_reg = {}
    print("=== 1. Synthetic benchmark_v2 (Unseen Test Split) ===")
    print(f"{'Pass ID':<26} | {'Regime':<14} | {'Kalman':>8} | {'SNN v2':>8} | {'Hybrid':>8} | {'Hyb Upd':>7}")
    print("-" * 86)

    with open(manifest_path, "r") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue

            kf_rmse, _, _ = eval_kalman_sim(row["events_h5"], row["ground_truth_csv"])
            frames, targets = load_pass_tensors(row["events_h5"], row["ground_truth_csv"])
            pred, spread, _ = infer_snn_with_spread(model, frames.numpy(), device)

            gt = targets.numpy()
            est_t = (np.arange(len(gt)) + 0.5) * DT_STEP

            snn_rmse = float(np.sqrt(np.mean(np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1]) ** 2)))
            hyb_x, hyb_y, hyb_upd = run_hybrid_from_snn_preds(est_t, pred[:, 0], pred[:, 1], spread)
            hyb_rmse = float(np.sqrt(np.mean(np.hypot(hyb_x - gt[:, 0], hyb_y - gt[:, 1]) ** 2)))

            reg = row["regime"]
            by_reg.setdefault(reg, []).append((kf_rmse, snn_rmse, hyb_rmse))
            print(
                f"{row['pass_id']:<26} | {reg:<14} | {kf_rmse:>8.2f} | {snn_rmse:>8.2f} | "
                f"{hyb_rmse:>8.2f} | {hyb_upd:>6.1f}%"
            )

    print("\n--- Synthetic Test Regime Summary (Mean RMSE px) ---")
    print(f"{'Regime':<16} | {'Kalman':>8} | {'SNN v2':>8} | {'Hybrid SNN+KF':>13}")
    print("-" * 55)
    for reg, vals in by_reg.items():
        print(
            f"{reg:<16} | {np.mean([v[0] for v in vals]):>8.2f} | "
            f"{np.mean([v[1] for v in vals]):>8.2f} | {np.mean([v[2] for v in vals]):>13.2f}"
        )


def benchmark_ebssa(h5_path: str, model, device: torch.device, recordings: list[str]):
    print("\n=== 2. Zero-Shot Real-Sky EBSSA (3 Single-Target DAVIS Passes) ===")
    print(f"{'Recording':<30} | {'Kalman':>9} | {'SNN v2':>9} | {'Hybrid':>9} | {'Hyb Med':>8} | {'Hyb Upd':>7}")
    print("-" * 88)

    for rec in recordings:
        data = load_ebssa_recording(h5_path, rec, obj_id=None, duration_s=2.0)
        kf_res = eval_kalman_ebssa(data)

        n_bins = int(round(data["duration_s"] / DT_STEP))
        frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)
        bin_idx = np.floor(data["t_ev"] / DT_STEP).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        b = bin_idx[valid]
        gx = np.clip((data["x_ev"][valid] * (GRID_W / data["x_max"])).astype(np.int64), 0, GRID_W - 1)
        gy = np.clip((data["y_ev"][valid] * (GRID_H / data["y_max"])).astype(np.int64), 0, GRID_H - 1)
        np.add.at(frames, (b, data["p_ev"][valid], gy, gx), 1.0)
        frames = np.log1p(frames)

        pred, spread, _ = infer_snn_with_spread(model, frames, device)
        sx = pred[:, 0] * (data["x_max"] / ORIG_W)
        sy = pred[:, 1] * (data["y_max"] / ORIG_H)
        sp = spread * np.hypot(data["x_max"] / ORIG_W, data["y_max"] / ORIG_H) / np.sqrt(2.0)
        est_t = (np.arange(n_bins) + 0.5) * DT_STEP

        ref_x = np.interp(est_t, data["gt_t"], data["gt_x"])
        ref_y = np.interp(est_t, data["gt_t"], data["gt_y"])
        snn_rmse = float(np.sqrt(np.mean(np.hypot(sx - ref_x, sy - ref_y) ** 2)))

        hyb_x, hyb_y, hyb_upd = run_hybrid_from_snn_preds(est_t, sx, sy, sp)
        hyb_err = np.hypot(hyb_x - ref_x, hyb_y - ref_y)
        hyb_rmse = float(np.sqrt(np.mean(hyb_err**2)))
        hyb_med = float(np.median(hyb_err))

        print(
            f"{rec:<30} | {kf_res['rmse']:>9.2f} | {snn_rmse:>9.2f} | "
            f"{hyb_rmse:>9.2f} | {hyb_med:>8.2f} | {hyb_upd:>6.1f}%"
        )

        out_png = f"results/hybrid_ebssa_{rec}.png"
        plt.figure(figsize=(8, 6))
        plt.plot(data["gt_x"], data["gt_y"], "k-", linewidth=2.5, label="EBSSA Ground Truth")
        plt.plot(kf_res["x"], kf_res["y"], "r--", alpha=0.65, label=f"Kalman ({kf_res['rmse']:.1f} px)")
        plt.plot(sx, sy, "b:", alpha=0.65, label=f"SNN v2 ({snn_rmse:.1f} px)")
        plt.plot(hyb_x, hyb_y, "g-", linewidth=2.0, label=f"Hybrid SNN+KF ({hyb_rmse:.1f} px)")
        plt.xlim(0, data["x_max"])
        plt.ylim(data["y_max"], 0)
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.title(f"Zero-Shot Real-Sky Comparison: {rec}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=0.85,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/snn_tracker_v2_best.pt", map_location=device, weights_only=True))
    model.eval()

    benchmark_synthetic("data/benchmark_v2/manifest.csv", model, device)
    benchmark_ebssa(
        "data/ebssa/labelled_ebssa.h5",
        model,
        device,
        [
            "20170214-20-58_22285_SL-16RB",
            "cosmos_1848_davis_td",
            "SL14RB_leap_1933_davis_td",
        ],
    )


if __name__ == "__main__":
    main()

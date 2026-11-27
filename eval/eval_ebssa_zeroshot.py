import argparse
import os
import sys
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("baselines"))
sys.path.insert(0, os.path.abspath("models"))
from kalman_tracker import ConstantVelocityKalman2D, extract_cluster_centroid
from snn_tracker import SpikingConvLIFTracker


SNN_ORIG_W, SNN_ORIG_H = 346, 260
GRID_W, GRID_H = 86, 65
DT_STEP = 0.005  # 5 ms bins


def load_ebssa_recording(h5_path: str, rec_name: str, obj_id: int | None, duration_s: float):
    with h5py.File(h5_path, "r") as f:
        g = f[rec_name]
        x_max = float(np.squeeze(g["xMax"][()]))
        y_max = float(np.squeeze(g["yMax"][()]))

        td_ts = np.squeeze(g["TD"]["ts"][:]).astype(np.float64)
        td_x = np.squeeze(g["TD"]["x"][:]).astype(np.float64)
        td_y = np.squeeze(g["TD"]["y"][:]).astype(np.float64)
        td_p = np.squeeze(g["TD"]["p"][:]).astype(np.float64)

        obj_ts = np.squeeze(g["Obj"]["ts"][:]).astype(np.float64)
        obj_x = np.squeeze(g["Obj"]["x"][:]).astype(np.float64)
        obj_y = np.squeeze(g["Obj"]["y"][:]).astype(np.float64)
        obj_ids = np.squeeze(g["Obj"]["id"][:]).astype(np.int64)

    # Auto-detect microseconds vs seconds
    if td_ts.max() > 1e4:
        td_ts *= 1e-6
    if obj_ts.max() > 1e4:
        obj_ts *= 1e-6

    # Pick the object with the most labelled samples if not specified
    if obj_id is None:
        uniq, counts = np.unique(obj_ids, return_counts=True)
        obj_id = int(uniq[np.argmax(counts)])

    mask_obj = obj_ids == obj_id
    gt_t = obj_ts[mask_obj]
    gt_x = obj_x[mask_obj]
    gt_y = obj_y[mask_obj]

    order_gt = np.argsort(gt_t)
    gt_t, gt_x, gt_y = gt_t[order_gt], gt_x[order_gt], gt_y[order_gt]

    t_start = float(gt_t[0])
    t_end = min(float(gt_t[-1]), t_start + duration_s)

    # Slice ground truth and events to [t_start, t_end]
    gt_win = (gt_t >= t_start) & (gt_t <= t_end)
    gt_t = gt_t[gt_win] - t_start
    gt_x = gt_x[gt_win]
    gt_y = gt_y[gt_win]

    ev_win = (td_ts >= t_start) & (td_ts <= t_end)
    t_ev = td_ts[ev_win] - t_start
    x_ev = td_x[ev_win]
    y_ev = td_y[ev_win]
    p_ev = (td_p[ev_win] > 0).astype(np.int64)

    order_ev = np.argsort(t_ev)
    t_ev, x_ev, y_ev, p_ev = t_ev[order_ev], x_ev[order_ev], y_ev[order_ev], p_ev[order_ev]

    return {
        "x_max": x_max,
        "y_max": y_max,
        "obj_id": obj_id,
        "duration_s": t_end - t_start,
        "t_ev": t_ev,
        "x_ev": x_ev,
        "y_ev": y_ev,
        "p_ev": p_ev,
        "gt_t": gt_t,
        "gt_x": gt_x,
        "gt_y": gt_y,
    }


def run_kalman(data):
    t_ev = data["t_ev"]
    ev = np.column_stack([t_ev * 1e6, data["x_ev"], data["y_ev"], data["p_ev"]])
    gt_t, gt_x, gt_y = data["gt_t"], data["gt_x"], data["gt_y"]

    bins = np.arange(0.0, data["duration_s"] + DT_STEP, DT_STEP)
    kf, first_det = None, None
    est_t, est_x, est_y = [], [], []
    updates, total_steps = 0, 0

    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i + 1]
        t_mid = 0.5 * (b0 + b1)
        idx0 = np.searchsorted(t_ev, b0, side="left")
        idx1 = np.searchsorted(t_ev, b1, side="left")
        slice_ev, slice_t = ev[idx0:idx1], t_ev[idx0:idx1]
        total_steps += 1

        if kf is None:
            prev_xy = (first_det[0], first_det[1]) if first_det is not None else None
            centroid, t_meas, _ = extract_cluster_centroid(slice_ev, slice_t, prev_xy=prev_xy, on_only=True)
            if centroid is not None:
                if first_det is None:
                    first_det = (centroid[0], centroid[1], t_meas)
                    est_t.append(t_meas); est_x.append(centroid[0]); est_y.append(centroid[1])
                    updates += 1
                else:
                    dt_init = max(1e-4, t_meas - first_det[2])
                    vx0 = (centroid[0] - first_det[0]) / dt_init
                    vy0 = (centroid[1] - first_det[1]) / dt_init
                    kf = ConstantVelocityKalman2D(centroid[0], centroid[1], t_meas, vx0=vx0, vy0=vy0)
                    est_t.append(t_meas); est_x.append(kf.state[0]); est_y.append(kf.state[1])
                    updates += 1
        else:
            kf.predict(t_mid)
            centroid, t_meas, _ = extract_cluster_centroid(
                slice_ev, slice_t, prev_xy=kf.state[:2], search_radius=35.0, on_only=True
            )
            if centroid is not None:
                kf.predict(t_meas)
                if kf.update(centroid[0], centroid[1], gate_radius=25.0):
                    updates += 1
                est_t.append(t_meas)
            else:
                est_t.append(t_mid)
            est_x.append(kf.state[0]); est_y.append(kf.state[1])

    est_t, est_x, est_y = np.array(est_t), np.array(est_x), np.array(est_y)
    ref_x = np.interp(est_t, gt_t, gt_x)
    ref_y = np.interp(est_t, gt_t, gt_y)
    err = np.hypot(est_x - ref_x, est_y - ref_y)
    return {
        "t": est_t,
        "x": est_x,
        "y": est_y,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median": float(np.median(err)),
        "upd_pct": 100.0 * updates / max(1, total_steps),
    }


def run_snn(data, ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_bins = int(round(data["duration_s"] / DT_STEP))
    frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)

    bin_idx = np.floor(data["t_ev"] / DT_STEP).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    b = bin_idx[valid]
    gx = np.clip((data["x_ev"][valid] * (GRID_W / data["x_max"])).astype(np.int64), 0, GRID_W - 1)
    gy = np.clip((data["y_ev"][valid] * (GRID_H / data["y_max"])).astype(np.int64), 0, GRID_H - 1)
    ch = data["p_ev"][valid]

    np.add.at(frames, (b, ch, gy, gx), 1.0)
    frames = np.log1p(frames)

    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=0.85,
        orig_w=SNN_ORIG_W,
        orig_h=SNN_ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        x_in = torch.from_numpy(frames).unsqueeze(0).to(device)
        pred, spk_rate = model(x_in)
        pred = pred.squeeze(0).cpu().numpy()

    # Rescale from SNN training frame (346x260) to EBSSA sensor frame (xMax x yMax)
    est_x = pred[:, 0] * (data["x_max"] / SNN_ORIG_W)
    est_y = pred[:, 1] * (data["y_max"] / SNN_ORIG_H)
    est_t = (np.arange(n_bins) + 0.5) * DT_STEP

    ref_x = np.interp(est_t, data["gt_t"], data["gt_x"])
    ref_y = np.interp(est_t, data["gt_t"], data["gt_y"])
    err = np.hypot(est_x - ref_x, est_y - ref_y)
    return {
        "t": est_t,
        "x": est_x,
        "y": est_y,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median": float(np.median(err)),
        "spk_pct": 100.0 * spk_rate.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot EBSSA real-sky benchmark: Kalman vs SNN.")
    parser.add_argument("--h5", default="data/ebssa/labelled_ebssa.h5")
    parser.add_argument("--recording", default="ISS_2_td")
    parser.add_argument("--obj_id", type=int, default=None)
    parser.add_argument("--duration_s", type=float, default=2.0)
    parser.add_argument("--ckpt", default="checkpoints/snn_tracker_v2_best.pt")
    args = parser.parse_args()

    data = load_ebssa_recording(args.h5, args.recording, args.obj_id, args.duration_s)
    print(
        f"Recording: {args.recording} | Sensor: {int(data['x_max'])}x{int(data['y_max'])} | "
        f"Obj ID: {data['obj_id']} | Window: {data['duration_s']:.2f}s | Events: {len(data['t_ev']):,}"
    )

    kf_res = run_kalman(data)
    snn_res = run_snn(data, args.ckpt)

    print("\n=== Zero-Shot Real-Sky EBSSA Results ===")
    print(f"{'Tracker':<18} | {'RMSE (px)':>9} | {'Median (px)':>11} | {'Activity':>16}")
    print("-" * 64)
    print(f"{'Kalman Baseline':<18} | {kf_res['rmse']:>9.2f} | {kf_res['median']:>11.2f} | {kf_res['upd_pct']:>10.1f}% upd")
    print(f"{'SNN v2 (Zero-Shot)':<18} | {snn_res['rmse']:>9.2f} | {snn_res['median']:>11.2f} | {snn_res['spk_pct']:>10.2f}% spk")

    # Save trajectory plot
    out_png = f"results/ebssa_zeroshot_{args.recording}.png"
    plt.figure(figsize=(8, 6))
    plt.plot(data["gt_x"], data["gt_y"], "k-", linewidth=2.5, label="EBSSA Ground Truth")
    plt.plot(kf_res["x"], kf_res["y"], "r--", alpha=0.8, label=f"Kalman (RMSE: {kf_res['rmse']:.1f} px)")
    plt.plot(snn_res["x"], snn_res["y"], "g-", alpha=0.85, label=f"SNN v2 Zero-Shot (RMSE: {snn_res['rmse']:.1f} px)")
    plt.xlim(0, data["x_max"])
    plt.ylim(data["y_max"], 0)
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title(f"Zero-Shot Real-Sky Tracking: {args.recording} (Obj {data['obj_id']}, {data['duration_s']:.1f}s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved trajectory plot to {out_png}")


if __name__ == "__main__":
    main()

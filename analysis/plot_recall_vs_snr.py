import csv, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("eval"))
sys.path.insert(0, os.path.abspath("baselines"))
sys.path.insert(0, os.path.abspath("models"))

from eval_mot_synth import load_synth_mot, run_mot_snn_kf
from eval_frame_mot import run_frame_proxy
from snn_tracker import SpikingConvLIFTracker


def per_target_density(data, bin_s=0.005, radius_px=10.0):
    t_ev, x_ev, y_ev = data["t_ev"], data["x_ev"], data["y_ev"]
    out = {}
    for tid, tr in data["gt_tracks"].items():
        t_gt, x_gt, y_gt = tr["t"], tr["x"], tr["y"]
        bins = np.arange(float(t_gt[0]), float(t_gt[-1]), bin_s)
        if len(bins) == 0:
            out[tid] = 0.0
            continue
        gx = np.interp(bins, t_gt, x_gt)
        gy = np.interp(bins, t_gt, y_gt)
        i0 = np.searchsorted(t_ev, bins)
        i1 = np.searchsorted(t_ev, bins + bin_s)
        cnts = []
        for i in range(len(bins)):
            a, b = i0[i], i1[i]
            if b > a:
                dx = x_ev[a:b] - gx[i]
                dy = y_ev[a:b] - gy[i]
                cnts.append(int(np.sum(dx * dx + dy * dy <= radius_px * radius_px)))
            else:
                cnts.append(0)
        out[tid] = float(np.mean(cnts))
    return out


def per_target_detected(preds, gt_tracks, radius_px=40.0):
    """Return dict {gt_tid: 1 if matched within radius_px else 0} via 1-to-1 Hungarian."""
    gids = list(gt_tracks.keys())
    if isinstance(preds, list): preds = {i: tr for i, tr in enumerate(preds)}
    pids = list(preds.keys())
    matched = {g: 0 for g in gids}
    if not gids or not pids:
        return matched

    cost = np.full((len(gids), len(pids)), 1e6)
    for i, gid in enumerate(gids):
        gt = gt_tracks[gid]
        for j, pid in enumerate(pids):
            pr = preds[pid]
            t_lo = max(gt["t"][0], pr["t"][0])
            t_hi = min(gt["t"][-1], pr["t"][-1])
            if t_hi <= t_lo:
                continue
            m = (pr["t"] >= t_lo) & (pr["t"] <= t_hi)
            if m.sum() == 0:
                continue
            gx = np.interp(pr["t"][m], gt["t"], gt["x"])
            gy = np.interp(pr["t"][m], gt["t"], gt["y"])
            cost[i, j] = float(np.sqrt(np.mean((pr["x"][m] - gx) ** 2 + (pr["y"][m] - gy) ** 2)))

    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        if cost[i, j] <= radius_px:
            matched[gids[i]] = 1
    return matched


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker().to(dev)
    ckpt = torch.load("checkpoints/snn_mot_best.pt", map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()

    with open("data/benchmark_mot/manifest.csv", "r") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("val", "test")]

    # Val-tuned locked configs
    trackers = {
        "SNN (0.60, 5.5ms)":       lambda d: run_mot_snn_kf(model, d, dev, gate_px=35.0, mode="sigmoid", prob_thresh=0.60),
        "30 fps proxy (t=2,c=6,h=4)":  lambda d: run_frame_proxy(d, fps=30,  pix_thr=2, min_cluster=6, min_hits=4),
        "60 fps proxy (t=2,c=2,h=6)":  lambda d: run_frame_proxy(d, fps=60,  pix_thr=2, min_cluster=2, min_hits=6),
        "200 fps proxy (t=1,c=4,h=4)": lambda d: run_frame_proxy(d, fps=200, pix_thr=1, min_cluster=4, min_hits=4),
    }

    records = {k: [] for k in trackers}
    for row in rows:
        data = load_synth_mot(row["events_h5"], row["ground_truth_csv"])
        dens = per_target_density(data)
        for name, fn in trackers.items():
            preds = fn(data)
            det = per_target_detected(preds, data["gt_tracks"], radius_px=40.0)
            for tid, d_val in dens.items():
                records[name].append((d_val, det[tid]))

    # Bin by target local events/bin
    edges = [1.5, 3.0, 4.5, 6.0, 9.5]
    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
    colors = ["#00d4ff", "#ef4444", "#fb7185", "#f59e0b"]
    markers = ["o", "X", "D", "^"]

    print(f"=== Recall@40px vs. Target Density (val+test, N={len(records['SNN (0.60, 5.5ms)'])} targets) ===")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    ax.axvspan(1.45, 2.87, color="#a855f7", alpha=0.14, label="Real EBSSA ISS_2_td band (1.45–2.87 ev/bin)")

    for (name, recs), col, mk in zip(records.items(), colors, markers):
        arr = np.array(recs)
        xs, ys = [], []
        row_str = []
        for i in range(len(edges) - 1):
            m = (arr[:, 0] >= edges[i]) & (arr[:, 0] < edges[i + 1])
            if m.sum() > 0:
                rec = 100.0 * arr[m, 1].mean()
                xs.append(centers[i]); ys.append(rec)
                row_str.append(f"[{edges[i]:.1f}-{edges[i+1]:.1f}]: {rec:5.1f}% (n={int(m.sum())})")
        print(f"{name:<28} | " + " | ".join(row_str))
        ax.plot(xs, ys, marker=mk, lw=2.2, ms=7, color=col, label=name)

    ax.set_xlabel("Target Local Event Density (events / 5 ms bin, r = 10 px)")
    ax.set_ylabel("Recall @ 40 px (%)")
    ax.set_title("Recall vs. Target Event Density (Val-Locked Trackers, val+test)")
    ax.set_ylim(-5, 108)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.2, loc="lower right")
    plt.tight_layout()
    plt.savefig("results/recall_vs_snr.png", dpi=200)
    print("Saved results/recall_vs_snr.png")


if __name__ == "__main__":
    main()

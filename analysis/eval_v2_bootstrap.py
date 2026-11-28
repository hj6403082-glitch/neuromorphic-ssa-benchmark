import csv, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))
sys.path.insert(0, os.path.abspath("baselines"))

from snn_tracker import SpikingConvLIFTracker
from eval_mot_iss import run_mot_snn_kf, ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP
from eval_mot_synth import load_synth_mot, eval_pass_metrics
from eval_frame_mot import run_frame_proxy

MANIFEST = "data/benchmark_mot_v2_test60/manifest.csv"
CKPT = "checkpoints/snn_mot_best.pt"
N_BOOT = 1000
BINS = [(0.75, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 9.5)]


def target_ev_per_bin(data, gt_tr, radius_px=8.0):
    t_ev, x_ev, y_ev = data["t_ev"], data["x_ev"], data["y_ev"]
    t_gt, x_gt, y_gt = gt_tr["t"], gt_tr["x"], gt_tr["y"]
    if len(t_gt) < 2:
        return 0.0
    t0, t1 = float(t_gt[0]), float(t_gt[-1])
    mask = (t_ev >= t0) & (t_ev <= t1)
    if not np.any(mask):
        return 0.0
    te, xe, ye = t_ev[mask], x_ev[mask], y_ev[mask]
    gx = np.interp(te, t_gt, x_gt)
    gy = np.interp(te, t_gt, y_gt)
    near = np.hypot(xe - gx, ye - gy) <= radius_px
    n_bins = max(1.0, (t1 - t0) / DT_STEP)
    return float(np.sum(near)) / n_bins


def match_targets_in_pass(preds, gt_tracks, radius_px=40.0):
    if isinstance(preds, dict):
        preds = list(preds.values())
    gt_ids = sorted(gt_tracks.keys())
    matched = {gid: 0 for gid in gt_ids}
    if not preds or not gt_ids:
        return matched
    cost = np.full((len(gt_ids), len(preds)), 1e6, dtype=np.float64)
    for gi, gid in enumerate(gt_ids):
        gtr = gt_tracks[gid]
        t0, t1 = gtr["t"][0], gtr["t"][-1]
        for pi, ptr in enumerate(preds):
            pt = np.asarray(ptr["t"], dtype=np.float64)
            ov = (pt >= t0) & (pt <= t1)
            if not np.any(ov):
                continue
            gx = np.interp(pt[ov], gtr["t"], gtr["x"])
            gy = np.interp(pt[ov], gtr["t"], gtr["y"])
            px = np.asarray(ptr["x"], dtype=np.float64)[ov]
            py = np.asarray(ptr["y"], dtype=np.float64)[ov]
            rmse = float(np.sqrt(np.mean((px - gx) ** 2 + (py - gy) ** 2)))
            cost[gi, pi] = rmse
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        if cost[r, c] <= radius_px:
            matched[gt_ids[r]] = 1
    return matched


def boot_ci_agg(pass_stats, rng, n_boot=N_BOOT):
    n = len(pass_stats)
    recs, precs, f1s, rmses = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tp = sum(pass_stats[i]["tp"] for i in idx)
        fp = sum(pass_stats[i]["fp"] for i in idx)
        gt = sum(pass_stats[i]["gt"] for i in idx)
        rm = [pass_stats[i]["rmse"] for i in idx if not np.isnan(pass_stats[i]["rmse"])]
        rec = tp / max(1, gt)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        recs.append(rec * 100)
        precs.append(prec * 100)
        f1s.append(f1 * 100)
        if rm:
            rmses.append(float(np.mean(rm)))
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return ci(recs), ci(precs), ci(f1s), (ci(rmses) if rmses else (float("nan"), float("nan")))


def boot_ci_bin(hits, rng, n_boot=N_BOOT):
    arr = np.asarray(hits, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr)) * 100.0
    means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) * 100.0 for _ in range(n_boot)]
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker(2, 32, 0.85, ORIG_W, ORIG_H, GRID_W, GRID_H).to(dev)
    ckpt = torch.load(CKPT, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    dataset = [load_synth_mot(r["events_h5"], r["ground_truth_csv"]) for r in rows]

    # Precompute target densities
    densities = []
    for data in dataset:
        densities.append({gid: target_ev_per_bin(data, tr) for gid, tr in data["gt_tracks"].items()})

    trackers = [
        ("SNN v2 (0.50, mm=20, mh=10)", "#00d4ff", "o",
         lambda d: run_mot_snn_kf(model, d, dev, gate_px=35.0, max_missed=20, min_hits_to_report=10, mode="sigmoid", prob_thresh=0.50)),
        ("30 fps proxy (t=2,c=4,h=6)", "#ef4444", "X",
         lambda d: run_frame_proxy(d, fps=30, pix_thr=2, min_cluster=4, min_hits=6)),
        ("60 fps proxy (t=1,c=6,h=6)", "#fb7185", "D",
         lambda d: run_frame_proxy(d, fps=60, pix_thr=1, min_cluster=6, min_hits=6)),
        ("200 fps proxy (t=2,c=2,h=2)", "#f59e0b", "^",
         lambda d: run_frame_proxy(d, fps=200, pix_thr=2, min_cluster=2, min_hits=2)),
    ]

    rng = np.random.default_rng(42)
    snr_curves = {}

    print(f"=== benchmark_mot_v2_test60 ({len(dataset)} passes, {sum(len(d['gt_tracks']) for d in dataset)} targets, 1000-resample 95% CI) ===")
    print(f"{'Tracker':<28} | {'R@40 [95% CI]':<19} | {'P@40 [95% CI]':<19} | {'F1@40 [95% CI]':<19} | {'FP/p':>5} | {'IDSW':>4} | {'RMSE':>6}")
    print("-" * 116)

    for name, color, marker, fn in trackers:
        pass_stats = []
        bin_hits = {b: [] for b in BINS}
        tot_tp = tot_fp = tot_gt = tot_idsw = 0
        rmses = []

        for data, dens in zip(dataset, densities):
            preds = fn(data)
            m = eval_pass_metrics(preds, data["gt_tracks"], radii=(40.0,))
            tp = m["radii"][40.0]["tp"]
            fp = m["radii"][40.0]["fp"]
            rmse = m["radii"][40.0]["rmse"]
            pass_stats.append({"tp": tp, "fp": fp, "gt": m["n_gt"], "rmse": rmse})
            tot_tp += tp; tot_fp += fp; tot_gt += m["n_gt"]; tot_idsw += m["idsw"]
            if not np.isnan(rmse):
                rmses.append(rmse)

            matched = match_targets_in_pass(preds, data["gt_tracks"], radius_px=40.0)
            for gid, hit in matched.items():
                evb = dens[gid]
                for lo, hi in BINS:
                    if lo <= evb < hi:
                        bin_hits[(lo, hi)].append(hit)
                        break

        rec = tot_tp / max(1, tot_gt) * 100
        prec = tot_tp / max(1, tot_tp + tot_fp) * 100
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        m_rmse = float(np.mean(rmses)) if rmses else float("nan")
        r_ci, p_ci, f_ci, _ = boot_ci_agg(pass_stats, rng)

        print(
            f"{name:<28} | {rec:5.1f} [{r_ci[0]:4.1f},{r_ci[1]:4.1f}] | "
            f"{prec:5.1f} [{p_ci[0]:4.1f},{p_ci[1]:4.1f}] | "
            f"{f1:5.1f} [{f_ci[0]:4.1f},{f_ci[1]:4.1f}] | "
            f"{tot_fp/len(dataset):5.2f} | {tot_idsw:4d} | {m_rmse:5.2f}px"
        )
        snr_curves[name] = (color, marker, {b: (len(bin_hits[b]), *boot_ci_bin(bin_hits[b], rng)) for b in BINS})

    print("\n=== Recall@40px vs. Target Event Density (95% Bootstrap CI) ===")
    header = f"{'Density Bin (ev/5ms)':<22} | {'n':>3} | " + " | ".join(f"{t[0].split(' (')[0]:<20}" for t in trackers)
    print(header)
    print("-" * len(header))
    for b in BINS:
        n_b = snr_curves[trackers[0][0]][2][b][0]
        cols = []
        for name, _, _, _ in trackers:
            _, mu, lo, hi = snr_curves[name][2][b]
            cols.append(f"{mu:5.1f}% [{lo:4.1f},{hi:4.1f}]")
        print(f"[{b[0]:4.2f}, {b[1]:4.2f})         | {n_b:3d} | " + " | ".join(f"{c:<20}" for c in cols))

    # Plot Recall vs SNR with 95% CI bands
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    xs = [0.5 * (lo + hi) for lo, hi in BINS]
    for name, color, marker, _ in trackers:
        _, _, bdict = snr_curves[name]
        ys = [bdict[b][1] for b in BINS]
        los = [bdict[b][2] for b in BINS]
        his = [bdict[b][3] for b in BINS]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2.2, label=name)
        ax.fill_between(xs, los, his, color=color, alpha=0.16)

    ax.axvspan(1.45, 2.87, color="#00ff88", alpha=0.10, label="ISS_2_td target band (1.45–2.87 ev/bin)")
    ax.set_xlabel("Target Event Density (events / 5 ms bin)", fontsize=11)
    ax.set_ylabel("Recall@40px (%)", fontsize=11)
    ax.set_title("Recall vs. Target Event Density — benchmark_mot_v2_test60 (N=60 passes, 95% Bootstrap CI)", fontsize=11.5)
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    plt.savefig("results/recall_vs_snr.png", dpi=160)
    print("\nSaved updated 95% CI plot to results/recall_vs_snr.png")


if __name__ == "__main__":
    main()

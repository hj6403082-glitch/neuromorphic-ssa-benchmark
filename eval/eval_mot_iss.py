import argparse
import os
import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, os.path.abspath("baselines"))
sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))

from kalman_tracker import ConstantVelocityKalman2D
from snn_tracker import SpikingConvLIFTracker
from train_snn import ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP


def load_iss_multi_object(h5_path: str, rec_name: str, duration_s: float = 2.0, start_offset_s: float = 0.0):
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

    if td_ts.max() > 1e4:
        td_ts *= 1e-6
    if obj_ts.max() > 1e4:
        obj_ts *= 1e-6

    t_base = float(np.min(obj_ts)) + start_offset_s
    t_end = t_base + duration_s

    ev_mask = (td_ts >= t_base) & (td_ts <= t_end)
    t_ev = td_ts[ev_mask] - t_base
    x_ev = td_x[ev_mask]
    y_ev = td_y[ev_mask]
    p_ev = (td_p[ev_mask] > 0).astype(np.int64)

    order = np.argsort(t_ev)
    t_ev, x_ev, y_ev, p_ev = t_ev[order], x_ev[order], y_ev[order], p_ev[order]

    gt_tracks = {}
    for oid in np.unique(obj_ids):
        m = (obj_ids == oid) & (obj_ts >= t_base) & (obj_ts <= t_end)
        if np.sum(m) >= 5:
            gt_t = obj_ts[m] - t_base
            gt_x = obj_x[m]
            gt_y = obj_y[m]
            idx = np.argsort(gt_t)
            gt_tracks[int(oid)] = {
                "t": gt_t[idx],
                "x": gt_x[idx],
                "y": gt_y[idx],
            }

    return {
        "x_max": x_max,
        "y_max": y_max,
        "duration_s": duration_s,
        "t_ev": t_ev,
        "x_ev": x_ev,
        "y_ev": y_ev,
        "p_ev": p_ev,
        "gt_tracks": gt_tracks,
    }


def extract_heatmap_peaks(
    probs_2d: np.ndarray,
    x_max: float,
    y_max: float,
    rel_thresh: float = 0.55,
    min_prob: float = 4e-3,
    nms_radius: int = 5,
    max_peaks: int = 8,
):
    """Extract prominent local maxima from an (H, W) probability map and map to (x_max, y_max)."""
    h, w = probs_2d.shape
    max_val = float(probs_2d.max())
    if max_val < min_prob:
        return []

    thresh = max(min_prob, rel_thresh * max_val)
    t_p = torch.from_numpy(probs_2d).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t_p, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)
    is_peak = (t_p == pooled) & (t_p >= thresh)

    ys, xs = torch.where(is_peak.squeeze(0).squeeze(0))
    vals = probs_2d[ys.numpy(), xs.numpy()]
    order = np.argsort(-vals)[:max_peaks]

    dets = []
    for idx in order:
        gx, gy = float(xs[idx]), float(ys[idx])
        sx = (gx / max(1, w - 1)) * x_max
        sy = (gy / max(1, h - 1)) * y_max
        dets.append((sx, sy, float(vals[idx])))
    return dets


def associate_detections(tracks, dets, gate_px: float = 35.0):
    """Match predicted track positions to detections within gate_px."""
    if not tracks or not dets:
        return [], list(range(len(dets)))

    cost = np.full((len(tracks), len(dets)), 1e6, dtype=np.float64)
    for ti, tr in enumerate(tracks):
        px, py = tr["kf"].state[0], tr["kf"].state[1]
        for di, (dx, dy, _) in enumerate(dets):
            dist = np.hypot(px - dx, py - dy)
            if dist <= gate_px:
                cost[ti, di] = dist

    matches = []
    matched_dets = set()

    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= gate_px:
                matches.append((r, c))
                matched_dets.add(c)
    else:
        pairs = []
        for r in range(len(tracks)):
            for c in range(len(dets)):
                if cost[r, c] <= gate_px:
                    pairs.append((cost[r, c], r, c))
        pairs.sort()
        used_r = set()
        for _, r, c in pairs:
            if r not in used_r and c not in matched_dets:
                used_r.add(r)
                matched_dets.add(c)
                matches.append((r, c))

    unmatched_dets = [di for di in range(len(dets)) if di not in matched_dets]
    return matches, unmatched_dets


@torch.no_grad()
def run_mot_snn_kf(
    model: SpikingConvLIFTracker,
    data: dict,
    device: torch.device,
    gate_px: float = 35.0,
    max_missed: int = 6,
    min_hits_to_report: int = 5,
    mode: str = "softmax",
    prob_thresh: float = 0.25,
):
    n_bins = int(round(data["duration_s"] / DT_STEP))
    frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)

    b_idx = np.floor(data["t_ev"] / DT_STEP).astype(np.int64)
    valid = (b_idx >= 0) & (b_idx < n_bins)
    gx = np.clip((data["x_ev"][valid] * (GRID_W / data["x_max"])).astype(np.int64), 0, GRID_W - 1)
    gy = np.clip((data["y_ev"][valid] * (GRID_H / data["y_max"])).astype(np.int64), 0, GRID_H - 1)
    np.add.at(frames, (b_idx[valid], data["p_ev"][valid], gy, gx), 1.0)
    frames = np.log1p(frames)

    x_seq = torch.from_numpy(frames).unsqueeze(0).to(device)
    B, T, _, H, W = x_seq.shape

    mem1 = torch.zeros(B, model.conv1.out_channels, H, W, device=device)
    mem2 = torch.zeros(B, model.conv2.out_channels, H, W, device=device)
    mem3 = torch.zeros(B, model.conv3.out_channels, H, W, device=device)

    tracks = []
    next_tid = 1
    finished_tracks = []

    for t in range(T):
        t_sec = (t + 0.5) * DT_STEP
        cur1 = model.bn1(model.conv1(x_seq[:, t]))
        spk1, mem1 = model.lif1(cur1, mem1)

        cur2 = model.bn2(model.conv2(spk1))
        spk2, mem2 = model.lif2(cur2, mem2)

        cur3 = model.bn3(model.conv3(spk2))
        spk3, mem3 = model.lif3(cur3, mem3)

        logits = model.head(spk3 + 0.25 * mem3).reshape(B, -1)
        if mode == "sigmoid":
            probs_2d = torch.sigmoid(logits).reshape(H, W).cpu().numpy()
            dets = extract_heatmap_peaks(
                probs_2d, data["x_max"], data["y_max"],
                rel_thresh=0.0, min_prob=prob_thresh, nms_radius=4, max_peaks=8
            )
        else:
            probs_2d = F.softmax(logits, dim=-1).reshape(H, W).cpu().numpy()
            dets = extract_heatmap_peaks(probs_2d, data["x_max"], data["y_max"])

        for tr in tracks:
            tr["kf"].predict(t_sec)

        matches, unmatched_dets = associate_detections(tracks, dets, gate_px=gate_px)
        matched_tids = set()

        for ti, di in matches:
            dx, dy, _ = dets[di]
            tr = tracks[ti]
            tr["kf"].update(dx, dy, gate_radius=gate_px)
            tr["missed"] = 0
            tr["hits"] += 1
            tr["t"].append(t_sec)
            tr["x"].append(float(tr["kf"].state[0]))
            tr["y"].append(float(tr["kf"].state[1]))
            matched_tids.add(ti)

        surviving = []
        for ti, tr in enumerate(tracks):
            if ti not in matched_tids:
                tr["missed"] += 1
                if tr["missed"] <= max_missed:
                    tr["t"].append(t_sec)
                    tr["x"].append(float(tr["kf"].state[0]))
                    tr["y"].append(float(tr["kf"].state[1]))
                    surviving.append(tr)
                else:
                    if tr["hits"] >= min_hits_to_report:
                        finished_tracks.append(tr)
            else:
                surviving.append(tr)
        tracks = surviving

        for di in unmatched_dets:
            dx, dy, _ = dets[di]
            tracks.append({
                "tid": next_tid,
                "kf": ConstantVelocityKalman2D(dx, dy, t_sec, vx0=0.0, vy0=0.0),
                "missed": 0,
                "hits": 1,
                "t": [t_sec],
                "x": [dx],
                "y": [dy],
            })
            next_tid += 1

    for tr in tracks:
        if tr["hits"] >= min_hits_to_report:
            finished_tracks.append(tr)

    for tr in finished_tracks:
        tr["t"] = np.array(tr["t"])
        tr["x"] = np.array(tr["x"])
        tr["y"] = np.array(tr["y"])

    return finished_tracks


def evaluate_against_gt(pred_tracks, gt_tracks: dict, match_thresh_px: float = 40.0):
    """Compute per-GT-object best matching RMSE and hit summary."""
    results = []
    hits = 0
    for oid, gt in sorted(gt_tracks.items()):
        best_rmse = None
        best_tid = None
        best_overlap = 0.0

        for tr in pred_tracks:
            t_lo = max(float(gt["t"][0]), float(tr["t"][0]))
            t_hi = min(float(gt["t"][-1]), float(tr["t"][-1]))
            if t_hi <= t_lo:
                continue
            eval_t = np.linspace(t_lo, t_hi, num=max(5, int((t_hi - t_lo) / DT_STEP)))
            gx_i = np.interp(eval_t, gt["t"], gt["x"])
            gy_i = np.interp(eval_t, gt["t"], gt["y"])
            px_i = np.interp(eval_t, tr["t"], tr["x"])
            py_i = np.interp(eval_t, tr["t"], tr["y"])
            rmse = float(np.sqrt(np.mean((px_i - gx_i) ** 2 + (py_i - gy_i) ** 2)))
            overlap_frac = (t_hi - t_lo) / max(1e-6, float(gt["t"][-1] - gt["t"][0]))

            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_tid = tr["tid"]
                best_overlap = overlap_frac

        matched = best_rmse is not None and best_rmse <= match_thresh_px
        if matched:
            hits += 1
        results.append({
            "oid": oid,
            "best_tid": best_tid,
            "rmse": best_rmse,
            "overlap": best_overlap,
            "matched": matched,
        })

    matched_rmses = [r["rmse"] for r in results if r["matched"]]
    summary = {
        "total_gt": len(results),
        "hits": hits,
        "pred_tracks": len(pred_tracks),
        "mean_matched_rmse": float(np.mean(matched_rmses)) if matched_rmses else float("nan"),
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Multi-object tracking evaluation on EBSSA ISS_2_td.")
    parser.add_argument("--h5", default="data/ebssa/labelled_ebssa.h5")
    parser.add_argument("--recording", default="ISS_2_td")
    parser.add_argument("--duration_s", type=float, default=2.0)
    parser.add_argument("--start_offset_s", type=float, default=0.0)
    parser.add_argument("--ckpt", default="checkpoints/snn_tracker_v2_best.pt")
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--gate_px", type=float, default=35.0)
    parser.add_argument("--mode", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--prob_thresh", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=args.beta,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    data = load_iss_multi_object(
        args.h5, args.recording, duration_s=args.duration_s, start_offset_s=args.start_offset_s
    )
    pred_tracks = run_mot_snn_kf(model, data, device, gate_px=args.gate_px, mode=args.mode, prob_thresh=args.prob_thresh)
    per_obj, summary = evaluate_against_gt(pred_tracks, data["gt_tracks"])

    print(
        f"Recording: {args.recording} | Sensor: {int(data['x_max'])}x{int(data['y_max'])} | "
        f"Offset: {args.start_offset_s:.2f}s | Window: {data['duration_s']:.2f}s | Events: {len(data['t_ev']):,}"
    )
    print(
        f"Active GT Objects in Window: {summary['total_gt']} | "
        f"Predicted Tracks: {summary['pred_tracks']} | "
        f"Matched (<40px): {summary['hits']}/{summary['total_gt']} | "
        f"Mean Matched RMSE: {summary['mean_matched_rmse']:.2f} px\n"
    )

    print(f"{'GT Obj ID':>10} | {'Best Track':>10} | {'Overlap':>8} | {'RMSE (px)':>10} | {'Status':<10}")
    print("-" * 60)
    for row in per_obj:
        rmse_str = f"{row['rmse']:10.2f}" if row["rmse"] is not None else "       N/A"
        tid_str = f"T{row['best_tid']:02d}" if row["best_tid"] is not None else "       -"
        status = "MATCHED" if row["matched"] else "UNMATCHED"
        print(f"{row['oid']:>10d} | {tid_str:>10} | {100*row['overlap']:>7.1f}% | {rmse_str} | {status:<10}")

    os.makedirs("results", exist_ok=True)
    out_png = f"results/mot_{args.recording}.png"
    plt.figure(figsize=(8, 6))
    for i, (oid, gt) in enumerate(sorted(data["gt_tracks"].items())):
        plt.plot(gt["x"], gt["y"], "k--", linewidth=1.2, alpha=0.65, label="GT Objects" if i == 0 else "")
    for tr in pred_tracks:
        plt.plot(tr["x"], tr["y"], linewidth=1.8, alpha=0.9, label=f"Track T{tr['tid']:02d}")
    plt.xlim(0, data["x_max"])
    plt.ylim(data["y_max"], 0)
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title(f"Multi-Object SNN+KF Tracking: {args.recording} ({summary['hits']}/{summary['total_gt']} matched)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved MOT plot to {out_png}")


if __name__ == "__main__":
    main()

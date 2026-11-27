import argparse, csv, os, sys
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("eval"))
from eval_mot_synth import load_synth_mot, eval_pass_metrics


def run_frame_mot_kf(data, fps, min_events=4, gate_px=35.0, min_hits=2):
    """Bin events into 1/fps frames, extract connected-component centroids, and link via nearest-neighbor KF."""
    dt = 1.0 / fps
    n_frames = int(np.floor(data["duration_s"] / dt))
    W, H = int(data["x_max"]), int(data["y_max"])
    t_ev, x_ev, y_ev = data["t_ev"], data["x_ev"], data["y_ev"]

    tracks = {}
    next_id = 0
    active = {}  # tid -> {"x", "y", "vx", "vy", "missed", "hits"}

    for f in range(n_frames):
        t0, t1 = f * dt, (f + 1) * dt
        t_mid = 0.5 * (t0 + t1)
        m = (t_ev >= t0) & (t_ev < t1)
        dets = []
        if m.sum() >= min_events:
            img = np.zeros((H, W), dtype=np.int32)
            xi = np.clip(x_ev[m].astype(int), 0, W - 1)
            yi = np.clip(y_ev[m].astype(int), 0, H - 1)
            np.add.at(img, (yi, xi), 1)
            # Threshold frame at >= 2 events/pixel (or >= 1 for fast 200 fps frames)
            thr = 1 if fps >= 100 else 2
            lbl, n_cc = label(img >= thr)
            if n_cc > 0:
                counts = np.bincount(lbl.ravel())
                for cid in range(1, n_cc + 1):
                    if counts[cid] >= min_events:
                        cy, cx = center_of_mass(img, lbl, cid)
                        dets.append((float(cx), float(cy)))

        # Predict active tracks forward by dt
        tids = list(active.keys())
        for tid in tids:
            active[tid]["x"] += active[tid]["vx"] * dt
            active[tid]["y"] += active[tid]["vy"] * dt
            active[tid]["missed"] += 1

        matched_d = set()
        if tids and dets:
            cost = np.full((len(tids), len(dets)), 1e6)
            for i, tid in enumerate(tids):
                for j, (dx, dy) in enumerate(dets):
                    cost[i, j] = np.hypot(active[tid]["x"] - dx, active[tid]["y"] - dy)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < gate_px:
                    tid = tids[i]
                    dx, dy = dets[j]
                    tr = active[tid]
                    # Alpha-beta velocity update
                    tr["vx"] = 0.5 * tr["vx"] + 0.5 * (dx - tr["x"]) / dt
                    tr["vy"] = 0.5 * tr["vy"] + 0.5 * (dy - tr["y"]) / dt
                    tr["x"], tr["y"] = dx, dy
                    tr["missed"] = 0
                    tr["hits"] += 1
                    tracks[tid]["t"].append(t_mid)
                    tracks[tid]["x"].append(dx)
                    tracks[tid]["y"].append(dy)
                    matched_d.add(j)

        for j, (dx, dy) in enumerate(dets):
            if j not in matched_d:
                tid = next_id
                next_id += 1
                active[tid] = {"x": dx, "y": dy, "vx": 0.0, "vy": 0.0, "missed": 0, "hits": 1}
                tracks[tid] = {"t": [t_mid], "x": [dx], "y": [dy]}

        for tid in list(active.keys()):
            if active[tid]["missed"] > 2:
                del active[tid]

    out = {}
    for tid, tr in tracks.items():
        if len(tr["t"]) >= min_hits:
            out[tid] = {"t": np.array(tr["t"]), "x": np.array(tr["x"]), "y": np.array(tr["y"])}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/benchmark_mot/manifest.csv")
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args()

    with open(args.manifest, "r") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    print(f"Event-Binned Frame Baseline | Split: {args.split.upper()} ({len(rows)} passes)")
    print(f"{'FPS':>5} | {'dt(ms)':>6} | {'R@5px':>6} | {'R@10px':>6} | {'R@40px':>6} | {'P@40px':>6} | {'FP/pass':>7} | {'IDSW':>4} | {'RMSE@40':>8}")
    print("-" * 80)
    for fps in (30, 60, 200):
        tot_gt = tot_pred = tot_idsw = 0
        agg = {r: {"tp": 0, "fp": 0, "rmses": []} for r in (5.0, 10.0, 40.0)}
        for row in rows:
            data = load_synth_mot(row["events_h5"], row["ground_truth_csv"])
            preds = run_frame_mot_kf(data, fps=fps)
            m = eval_pass_metrics(preds, data["gt_tracks"], radii=(5.0, 10.0, 40.0))
            tot_gt += m["n_gt"]; tot_pred += m["n_pred"]; tot_idsw += m["idsw"]
            for r in (5.0, 10.0, 40.0):
                agg[r]["tp"] += m["radii"][r]["tp"]
                agg[r]["fp"] += m["radii"][r]["fp"]
                if not np.isnan(m["radii"][r]["rmse"]):
                    agg[r]["rmses"].append(m["radii"][r]["rmse"])
        n_p = max(1, len(rows))
        r5 = 100 * agg[5.0]["tp"] / max(1, tot_gt)
        r10 = 100 * agg[10.0]["tp"] / max(1, tot_gt)
        r40 = 100 * agg[40.0]["tp"] / max(1, tot_gt)
        p40 = 100 * agg[40.0]["tp"] / max(1, tot_pred)
        fp_p = agg[40.0]["fp"] / n_p
        rmse = float(np.mean(agg[40.0]["rmses"])) if agg[40.0]["rmses"] else float("nan")
        print(f"{fps:5d} | {1000/fps:6.1f} | {r5:5.1f}% | {r10:5.1f}% | {r40:5.1f}% | {p40:5.1f}% | {fp_p:7.2f} | {tot_idsw:4d} | {rmse:6.2f} px")


if __name__ == "__main__":
    main()

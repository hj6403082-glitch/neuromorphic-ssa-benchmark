import argparse, csv, os, sys
import numpy as np
from scipy.ndimage import label, center_of_mass, uniform_filter
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("eval"))
from eval_mot_synth import load_synth_mot, eval_pass_metrics


def run_frame_proxy(data, fps, pix_thr=2, min_cluster=3, min_hits=3, gate_px=35.0, max_missed=2):
    """Event-binned frame proxy: 1/fps binning + 3x3 neighborhood noise filter + CC centroiding + KF."""
    dt = 1.0 / fps
    n_frames = int(np.floor(data["duration_s"] / dt))
    W, H = int(data["x_max"]), int(data["y_max"])
    t_ev, x_ev, y_ev = data["t_ev"], data["x_ev"], data["y_ev"]

    tracks, active, next_id = {}, {}, 0

    for f in range(n_frames):
        t0, t1 = f * dt, (f + 1) * dt
        t_mid = 0.5 * (t0 + t1)
        m = (t_ev >= t0) & (t_ev < t1)
        dets = []
        if m.sum() >= min_cluster:
            img = np.zeros((H, W), dtype=np.float32)
            xi = np.clip(x_ev[m].astype(int), 0, W - 1)
            yi = np.clip(y_ev[m].astype(int), 0, H - 1)
            np.add.at(img, (yi, xi), 1.0)

            # 3x3 neighborhood sum rejects isolated hot/background pixels
            nbr_sum = uniform_filter(img, size=3, mode="constant") * 9.0
            mask = (img >= pix_thr) & (nbr_sum >= (pix_thr + 2.0))
            lbl, n_cc = label(mask)
            if n_cc > 0:
                counts = np.bincount(lbl.ravel())
                for cid in range(1, n_cc + 1):
                    if counts[cid] >= min_cluster:
                        cy, cx = center_of_mass(img, lbl, cid)
                        dets.append((float(cx), float(cy)))

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
                    tr["vx"] = 0.5 * tr["vx"] + 0.5 * (dx - tr["x"]) / dt
                    tr["vy"] = 0.5 * tr["vy"] + 0.5 * (dy - tr["y"]) / dt
                    tr["x"], tr["y"], tr["missed"] = dx, dy, 0
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
            if active[tid]["missed"] > max_missed:
                del active[tid]

    return {
        tid: {"t": np.array(tr["t"]), "x": np.array(tr["x"]), "y": np.array(tr["y"])}
        for tid, tr in tracks.items()
        if len(tr["t"]) >= min_hits
    }


def eval_split(passes, fps, cfg):
    tot_gt = tot_pred = tot_idsw = 0
    agg = {r: {"tp": 0, "fp": 0, "rmses": []} for r in (5.0, 10.0, 40.0)}
    for data in passes:
        preds = run_frame_proxy(data, fps=fps, **cfg)
        m = eval_pass_metrics(preds, data["gt_tracks"], radii=(5.0, 10.0, 40.0))
        tot_gt += m["n_gt"]; tot_pred += m["n_pred"]; tot_idsw += m["idsw"]
        for r in (5.0, 10.0, 40.0):
            agg[r]["tp"] += m["radii"][r]["tp"]
            agg[r]["fp"] += m["radii"][r]["fp"]
            if not np.isnan(m["radii"][r]["rmse"]):
                agg[r]["rmses"].append(m["radii"][r]["rmse"])
    n_p = max(1, len(passes))
    r5 = 100.0 * agg[5.0]["tp"] / max(1, tot_gt)
    r10 = 100.0 * agg[10.0]["tp"] / max(1, tot_gt)
    r40 = 100.0 * agg[40.0]["tp"] / max(1, tot_gt)
    p40 = 100.0 * agg[40.0]["tp"] / max(1, tot_pred)
    f1 = (2 * p40 * r40 / (p40 + r40)) if (p40 + r40) > 0 else 0.0
    rmse = float(np.mean(agg[40.0]["rmses"])) if agg[40.0]["rmses"] else float("nan")
    return {"r5": r5, "r10": r10, "r40": r40, "p40": p40, "f1": f1,
            "fp_pass": agg[40.0]["fp"] / n_p, "idsw": tot_idsw, "rmse": rmse}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/benchmark_mot/manifest.csv")
    args = p.parse_args()

    with open(args.manifest, "r") as f:
        rows = list(csv.DictReader(f))
    val_passes = [load_synth_mot(r["events_h5"], r["ground_truth_csv"]) for r in rows if r["split"] == "val"]
    test_passes = [load_synth_mot(r["events_h5"], r["ground_truth_csv"]) for r in rows if r["split"] == "test"]

    grid = [
        {"pix_thr": pt, "min_cluster": mc, "min_hits": mh}
        for pt in (1, 2, 3)
        for mc in (2, 4, 6)
        for mh in (2, 4, 6)
    ]

    print(f"Event-Binned Frame Proxy (Val-Tuned -> Test-Locked | {len(val_passes)} val / {len(test_passes)} test)")
    print(f"{'FPS':>4} | {'Best Val Config':<22} | {'Split':<4} | {'R@5':>5} | {'R@40':>5} | {'P@40':>5} | {'F1@40':>5} | {'FP/p':>5} | {'IDSW':>4} | {'RMSE':>7}")
    print("-" * 92)
    for fps in (30, 60, 200):
        best_cfg, best_val = None, None
        for cfg in grid:
            res = eval_split(val_passes, fps, cfg)
            if best_val is None or res["f1"] > best_val["f1"]:
                best_val, best_cfg = res, cfg
        test_res = eval_split(test_passes, fps, best_cfg)
        cfg_str = f"t={best_cfg['pix_thr']},c={best_cfg['min_cluster']},h={best_cfg['min_hits']}"
        for sp, r in [("VAL", best_val), ("TEST", test_res)]:
            print(f"{fps:4d} | {cfg_str:<22} | {sp:<4} | {r['r5']:4.1f}% | {r['r40']:4.1f}% | {r['p40']:4.1f}% | {r['f1']:5.1f} | {r['fp_pass']:5.2f} | {r['idsw']:4d} | {r['rmse']:5.2f}px")
        print("-" * 92)


if __name__ == "__main__":
    main()

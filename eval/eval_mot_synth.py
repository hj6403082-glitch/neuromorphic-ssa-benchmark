import argparse, csv, os, sys, h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))
from snn_tracker import SpikingConvLIFTracker
from eval_mot_iss import run_mot_snn_kf, ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP


def load_synth_mot(h5_path, gt_csv, duration_s=2.0):
    with h5py.File(h5_path, "r") as f:
        t_ev, x_ev, y_ev, p_ev = f["t"][:], f["x"][:], f["y"][:], f["p"][:].astype(np.int64)
    gt_tracks = {}
    with open(gt_csv, "r") as f:
        for row in csv.DictReader(f):
            oid = int(row["id"])
            gt_tracks.setdefault(oid, {"t": [], "x": [], "y": []})
            gt_tracks[oid]["t"].append(float(row["t"]))
            gt_tracks[oid]["x"].append(float(row["x"]))
            gt_tracks[oid]["y"].append(float(row["y"]))
    for oid in gt_tracks:
        for k in ("t", "x", "y"):
            gt_tracks[oid][k] = np.array(gt_tracks[oid][k])
    return {
        "x_max": float(ORIG_W), "y_max": float(ORIG_H), "duration_s": duration_s,
        "t_ev": t_ev, "x_ev": x_ev, "y_ev": y_ev, "p_ev": p_ev, "gt_tracks": gt_tracks,
    }


def eval_pass_metrics(pred_tracks, gt_tracks, radii=(5.0, 10.0, 40.0)):
    """Compute 1-to-1 track-level precision/recall/FP and frame-level ID switches."""
    if isinstance(pred_tracks, list):
        norm = {}
        for idx, tr in enumerate(pred_tracks):
            if isinstance(tr, dict):
                tid = tr.get("id", tr.get("track_id", idx))
                norm[tid] = {"t": np.asarray(tr["t"]), "x": np.asarray(tr["x"]), "y": np.asarray(tr["y"])}
            else:
                tid = getattr(tr, "id", getattr(tr, "track_id", idx))
                norm[tid] = {"t": np.asarray(tr.t), "x": np.asarray(tr.x), "y": np.asarray(tr.y)}
        pred_tracks = norm
    gt_ids = sorted(gt_tracks.keys())
    pred_ids = sorted(pred_tracks.keys())
    n_gt, n_pred = len(gt_ids), len(pred_ids)

    # Precompute pairwise track RMSE over overlapping timestamps
    BIG = 1e6
    cost = np.full((n_gt, n_pred), BIG, dtype=np.float64)
    for i, gid in enumerate(gt_ids):
        gt = gt_tracks[gid]
        for j, pid in enumerate(pred_ids):
            pt = pred_tracks[pid]
            if len(pt["t"]) == 0:
                continue
            t_min, t_max = max(gt["t"][0], pt["t"][0]), min(gt["t"][-1], pt["t"][-1])
            mask = (pt["t"] >= t_min) & (pt["t"] <= t_max)
            if mask.sum() < 2:
                continue
            gx = np.interp(pt["t"][mask], gt["t"], gt["x"])
            gy = np.interp(pt["t"][mask], gt["t"], gt["y"])
            cost[i, j] = float(np.sqrt(np.mean((pt["x"][mask] - gx) ** 2 + (pt["y"][mask] - gy) ** 2)))

    row_ind, col_ind = linear_sum_assignment(cost) if (n_gt and n_pred) else ([], [])
    out = {"n_gt": n_gt, "n_pred": n_pred, "radii": {}}
    for r in radii:
        matched_rmses = [cost[i, j] for i, j in zip(row_ind, col_ind) if cost[i, j] < r]
        tp = len(matched_rmses)
        out["radii"][r] = {
            "tp": tp,
            "fp": n_pred - tp,
            "rmse": float(np.mean(matched_rmses)) if matched_rmses else float("nan"),
        }

    # Frame-level ID switches (IDSW) at largest radius (40 px)
    r_idsw = max(radii)
    n_bins = int(round(2.0 / DT_STEP))
    last_match = {}
    idsw = 0
    for b in range(n_bins):
        tb = (b + 0.5) * DT_STEP
        act_g, act_p = [], []
        for gid in gt_ids:
            gt = gt_tracks[gid]
            if gt["t"][0] <= tb <= gt["t"][-1]:
                act_g.append((gid, np.interp(tb, gt["t"], gt["x"]), np.interp(tb, gt["t"], gt["y"])))
        for pid in pred_ids:
            pt = pred_tracks[pid]
            if len(pt["t"]) and pt["t"][0] <= tb <= pt["t"][-1]:
                act_p.append((pid, np.interp(tb, pt["t"], pt["x"]), np.interp(tb, pt["t"], pt["y"])))
        if not act_g or not act_p:
            continue
        dmat = np.zeros((len(act_g), len(act_p)))
        for i, (_, gx, gy) in enumerate(act_g):
            for j, (_, px, py) in enumerate(act_p):
                dmat[i, j] = np.hypot(gx - px, gy - py)
        ri, ci = linear_sum_assignment(dmat)
        for i, j in zip(ri, ci):
            if dmat[i, j] < r_idsw:
                gid, pid = act_g[i][0], act_p[j][0]
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                last_match[gid] = pid
    out["idsw"] = idsw
    return out


def evaluate_split(model, dev, rows, prob_thresh, gate_px=35.0):
    radii = (5.0, 10.0, 40.0)
    tot_gt = tot_pred = tot_idsw = 0
    agg = {r: {"tp": 0, "fp": 0, "rmses": []} for r in radii}
    for row in rows:
        data = load_synth_mot(row["events_h5"], row["ground_truth_csv"])
        preds = run_mot_snn_kf(model, data, dev, gate_px=gate_px, mode="sigmoid", prob_thresh=prob_thresh)
        m = eval_pass_metrics(preds, data["gt_tracks"], radii=radii)
        tot_gt += m["n_gt"]; tot_pred += m["n_pred"]; tot_idsw += m["idsw"]
        for r in radii:
            agg[r]["tp"] += m["radii"][r]["tp"]
            agg[r]["fp"] += m["radii"][r]["fp"]
            if not np.isnan(m["radii"][r]["rmse"]):
                agg[r]["rmses"].append(m["radii"][r]["rmse"])
    n_p = max(1, len(rows))
    res = {"thresh": prob_thresh, "n_gt": tot_gt, "n_pred": tot_pred, "idsw": tot_idsw, "n_passes": n_p}
    for r in radii:
        tp, fp = agg[r]["tp"], agg[r]["fp"]
        rec = tp / max(1, tot_gt)
        prec = tp / max(1, tot_pred)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        res[r] = {
            "tp": tp, "rec": rec, "prec": prec, "f1": f1,
            "fp_per_pass": fp / n_p,
            "rmse": float(np.mean(agg[r]["rmses"])) if agg[r]["rmses"] else float("nan"),
        }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/benchmark_mot/manifest.csv")
    p.add_argument("--ckpt", default="checkpoints/snn_mot_best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--prob_thresh", type=float, default=0.50)
    p.add_argument("--sweep", type=float, nargs="+", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker(2, 32, 0.85, ORIG_W, ORIG_H, GRID_W, GRID_H).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    with open(args.manifest, "r") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    thresholds = args.sweep if args.sweep else [args.prob_thresh]
    print(f"Split: {args.split.upper()} ({len(rows)} passes)")
    print(f"{'Thresh':>6} | {'R@5px':>6} | {'R@10px':>6} | {'R@40px':>6} | {'P@40px':>6} | {'F1@40':>6} | {'FP/pass':>7} | {'IDSW':>4} | {'RMSE@40':>8}")
    print("-" * 82)
    for th in thresholds:
        r = evaluate_split(model, dev, rows, th)
        print(
            f"{th:6.2f} | {100*r[5.0]['rec']:5.1f}% | {100*r[10.0]['rec']:5.1f}% | "
            f"{100*r[40.0]['rec']:5.1f}% | {100*r[40.0]['prec']:5.1f}% | {100*r[40.0]['f1']:6.3f} | "
            f"{r[40.0]['fp_per_pass']:7.2f} | {r['idsw']:4d} | {r[40.0]['rmse']:6.2f} px"
        )


if __name__ == "__main__":
    main()

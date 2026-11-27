import csv, os, sys, h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))
from snn_tracker import SpikingConvLIFTracker
from eval_mot_iss import run_mot_snn_kf, evaluate_against_gt, ORIG_W, ORIG_H, GRID_W, GRID_H


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


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingConvLIFTracker(2, 32, 0.85, ORIG_W, ORIG_H, GRID_W, GRID_H).to(dev)
    model.load_state_dict(torch.load("checkpoints/snn_mot_best.pt", map_location=dev, weights_only=True))
    model.eval()

    tot_gt = tot_hits = tot_pred = 0
    rmses = []
    print(f"{'Pass ID':<24} | {'GT':>2} | {'Pred':>4} | {'Match':>5} | {'Mean RMSE':>10}")
    print("-" * 58)
    with open("data/benchmark_mot/manifest.csv", "r") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            data = load_synth_mot(row["events_h5"], row["ground_truth_csv"])
            preds = run_mot_snn_kf(model, data, dev, gate_px=35.0, mode="sigmoid", prob_thresh=0.35)
            _, s = evaluate_against_gt(preds, data["gt_tracks"], match_thresh_px=40.0)
            tot_gt += s["total_gt"]; tot_hits += s["hits"]; tot_pred += s["pred_tracks"]
            if not np.isnan(s["mean_matched_rmse"]):
                rmses.append(s["mean_matched_rmse"])
            r_str = f"{s['mean_matched_rmse']:7.2f} px" if not np.isnan(s["mean_matched_rmse"]) else "       N/A"
            print(f"{row['pass_id']:<24} | {s['total_gt']:>2d} | {s['pred_tracks']:>4d} | {s['hits']:>2d}/{s['total_gt']:<2d} | {r_str}")
    print("-" * 58)
    print(f"TOTAL: {tot_hits}/{tot_gt} matched ({100*tot_hits/max(1,tot_gt):.1f}% recall) | Pred tracks: {tot_pred} | Mean RMSE: {np.mean(rmses) if rmses else float('nan'):.2f} px")


if __name__ == "__main__":
    main()

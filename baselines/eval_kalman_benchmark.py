import csv
import os
import sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(__file__))
from kalman_tracker import ConstantVelocityKalman2D, extract_cluster_centroid, load_ground_truth


def evaluate_pass(h5_path: str, gt_path: str, dt_step: float = 0.005):
    with h5py.File(h5_path, "r") as f:
        ev = f["events"][:]

    t_ev = ev[:, 0].astype(np.float64) * 1e-6
    order = np.argsort(t_ev)
    ev, t_ev = ev[order], t_ev[order]
    gt_t, gt_x, gt_y = load_ground_truth(gt_path)

    bins = np.arange(float(t_ev[0]), float(t_ev[-1]) + dt_step, dt_step)
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

    if not est_t:
        return np.nan, 0.0, len(ev)

    est_t, est_x, est_y = np.array(est_t), np.array(est_x), np.array(est_y)
    ref_x = np.interp(est_t, gt_t, gt_x)
    ref_y = np.interp(est_t, gt_t, gt_y)
    err_px = np.hypot(est_x - ref_x, est_y - ref_y)
    rmse_px = float(np.sqrt(np.mean(err_px**2)))
    upd_pct = 100.0 * updates / max(1, total_steps)
    return rmse_px, upd_pct, len(ev)


def main():
    manifest_path = "data/benchmark_v1/manifest.csv"
    by_regime = {}

    print(f"{'Pass ID':<30} | {'Regime':<15} | {'Speed':>7} | {'Events':>8} | {'RMSE (px)':>9} | {'Upd %':>6}")
    print("-" * 88)

    with open(manifest_path, "r") as f:
        for row in csv.DictReader(f):
            rmse, upd, n_ev = evaluate_pass(row["events_h5"], row["ground_truth_csv"])
            reg = row["regime"]
            by_regime.setdefault(reg, []).append((rmse, upd))
            print(
                f"{row['pass_id']:<30} | {reg:<15} | {float(row['speed_px_s']):>7.1f} | "
                f"{n_ev:>8d} | {rmse:>9.2f} | {upd:>5.1f}%"
            )

    print("\n=== Regime Summary ===")
    for reg, vals in by_regime.items():
        rmses = [v[0] for v in vals if not np.isnan(v[0])]
        upds = [v[1] for v in vals]
        print(f"{reg:<16} | Mean RMSE: {np.mean(rmses):6.2f} px | Mean Updates: {np.mean(upds):5.1f}%")


if __name__ == "__main__":
    main()

import csv
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def infer_time_scale(ts: np.ndarray) -> float:
    """Infer multiplier to convert timestamps to seconds (handles us, ms, or s)."""
    span = float(np.max(ts) - np.min(ts))
    if span > 1e5:
        return 1e-6  # microseconds
    if span > 1e3:
        return 1e-3  # milliseconds
    return 1.0       # seconds


def analyze_ebssa_entry(h5_path: str, key: str = "ISS_2_td"):
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            key = list(f.keys())[0]
        g = f[key]

        x_max = int(np.squeeze(g["xMax"][:]))
        y_max = int(np.squeeze(g["yMax"][:]))

        ev_x = np.squeeze(g["TD/x"][:]).astype(np.float64)
        ev_y = np.squeeze(g["TD/y"][:]).astype(np.float64)
        ev_p = np.squeeze(g["TD/p"][:]).astype(np.float64)
        ev_ts_raw = np.squeeze(g["TD/ts"][:]).astype(np.float64)

        obj_x = np.squeeze(g["Obj/x"][:]).astype(np.float64)
        obj_y = np.squeeze(g["Obj/y"][:]).astype(np.float64)
        obj_ts_raw = np.squeeze(g["Obj/ts"][:]).astype(np.float64)
        obj_id = np.squeeze(g["Obj/id"][:]).astype(np.float64)

    scale = infer_time_scale(ev_ts_raw)
    t0 = float(np.min(ev_ts_raw))
    ev_t = (ev_ts_raw - t0) * scale
    obj_t = (obj_ts_raw - t0) * scale

    # Focus on primary object ID (most frequent label)
    ids, counts = np.unique(obj_id, return_counts=True)
    main_id = ids[np.argmax(counts)]
    m = obj_id == main_id
    obj_t, obj_x, obj_y = obj_t[m], obj_x[m], obj_y[m]
    order = np.argsort(obj_t)
    obj_t, obj_x, obj_y = obj_t[order], obj_x[order], obj_y[order]

    # Restrict events to labeled object time window
    t_start, t_end = float(obj_t[0]), float(obj_t[-1])
    win = (ev_t >= t_start) & (ev_t <= t_end)
    ev_t, ev_x, ev_y, ev_p = ev_t[win], ev_x[win], ev_y[win], ev_p[win]

    dur = max(1e-6, t_end - t_start)
    ev_rate = len(ev_t) / dur
    on_ratio = float(np.mean(ev_p > 0))

    # Target apparent speed (median px/s)
    dt = np.diff(obj_t)
    valid = dt > 1e-4
    speed = np.hypot(np.diff(obj_x)[valid], np.diff(obj_y)[valid]) / dt[valid]
    med_speed = float(np.median(speed)) if len(speed) > 0 else 0.0

    # Spatial spread of events within 25 px of ground-truth target
    ref_x = np.interp(ev_t, obj_t, obj_x)
    ref_y = np.interp(ev_t, obj_t, obj_y)
    radial = np.hypot(ev_x - ref_x, ev_y - ref_y)
    near_radial = radial[radial <= 25.0]
    target_ev_rate = len(near_radial) / dur

    return {
        "name": f"EBSSA ({key})",
        "res": f"{x_max}x{y_max}",
        "duration_s": dur,
        "total_events": len(ev_t),
        "ev_rate": ev_rate,
        "target_ev_rate": target_ev_rate,
        "on_ratio": on_ratio,
        "med_speed_px_s": med_speed,
        "radial_near": near_radial,
        "radial_med": float(np.median(near_radial)) if len(near_radial) else np.nan,
        "radial_p90": float(np.percentile(near_radial, 90)) if len(near_radial) else np.nan,
    }


def analyze_synth(h5_path: str, gt_path: str):
    with h5py.File(h5_path, "r") as f:
        ev = f["events"][:]

    ev_t = ev[:, 0].astype(np.float64) * 1e-6
    ev_x = ev[:, 1].astype(np.float64)
    ev_y = ev[:, 2].astype(np.float64)
    ev_p = ev[:, 3].astype(np.float64)

    gt_t, gt_x, gt_y = [], [], []
    with open(gt_path, "r") as f:
        for row in csv.DictReader(f):
            gt_t.append(float(row["timestamp_s"]))
            gt_x.append(float(row["x"]))
            gt_y.append(float(row["y"]))
    gt_t, gt_x, gt_y = np.array(gt_t), np.array(gt_x), np.array(gt_y)

    dur = max(1e-6, float(ev_t[-1] - ev_t[0]))
    ev_rate = len(ev_t) / dur
    on_ratio = float(np.mean(ev_p > 0))

    dt = np.diff(gt_t)
    speed = np.hypot(np.diff(gt_x), np.diff(gt_y)) / np.maximum(dt, 1e-6)
    med_speed = float(np.median(speed))

    ref_x = np.interp(ev_t, gt_t, gt_x)
    ref_y = np.interp(ev_t, gt_t, gt_y)
    radial = np.hypot(ev_x - ref_x, ev_y - ref_y)
    near_radial = radial[radial <= 25.0]
    target_ev_rate = len(near_radial) / dur

    return {
        "name": "Synthetic v2e (ISS)",
        "res": "346x260",
        "duration_s": dur,
        "total_events": len(ev_t),
        "ev_rate": ev_rate,
        "target_ev_rate": target_ev_rate,
        "on_ratio": on_ratio,
        "med_speed_px_s": med_speed,
        "radial_near": near_radial,
        "radial_med": float(np.median(near_radial)) if len(near_radial) else np.nan,
        "radial_p90": float(np.percentile(near_radial, 90)) if len(near_radial) else np.nan,
    }


def main():
    real = analyze_ebssa_entry("data/ebssa/labelled_ebssa.h5", key="ISS_2_td")
    synth = analyze_synth("data/synth_iss_test/events.h5", "data/synth_iss_test/ground_truth.csv")

    print("=== Real-Sky (EBSSA) vs. Synthetic (v2e) Validation ===")
    print(f"{'Metric':<26} | {real['name']:<22} | {synth['name']:<22}")
    print("-" * 76)
    print(f"{'Resolution':<26} | {real['res']:<22} | {synth['res']:<22}")
    print(f"{'Labeled Duration (s)':<26} | {real['duration_s']:<22.2f} | {synth['duration_s']:<22.2f}")
    print(f"{'Total Event Rate (ev/s)':<26} | {real['ev_rate']:<22.1f} | {synth['ev_rate']:<22.1f}")
    print(f"{'Target Event Rate (ev/s)':<26} | {real['target_ev_rate']:<22.1f} | {synth['target_ev_rate']:<22.1f}")
    print(f"{'ON Polarity Ratio':<26} | {real['on_ratio']:<22.3f} | {synth['on_ratio']:<22.3f}")
    print(f"{'Median Speed (px/s)':<26} | {real['med_speed_px_s']:<22.1f} | {synth['med_speed_px_s']:<22.1f}")
    print(f"{'Spatial Spread Med (px)':<26} | {real['radial_med']:<22.2f} | {synth['radial_med']:<22.2f}")
    print(f"{'Spatial Spread P90 (px)':<26} | {real['radial_p90']:<22.2f} | {synth['radial_p90']:<22.2f}")

    # Plot radial spatial spread comparison
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, 25, 40)
    ax.hist(real["radial_near"], bins=bins, density=True, alpha=0.55, color="#38bdf8", label=real["name"])
    ax.hist(synth["radial_near"], bins=bins, density=True, alpha=0.55, color="#f59e0b", label=synth["name"])
    ax.set_title("Target Event Spatial Spread: Real EBSSA vs. Synthetic v2e")
    ax.set_xlabel("Radial Distance from Ground-Truth Center (px)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_plot = "docs/ebssa_vs_synth_validation.png"
    plt.savefig(out_plot, dpi=150)
    print(f"\nSaved validation plot to {out_plot}")


if __name__ == "__main__":
    main()

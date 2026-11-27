import csv
import time
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ConstantVelocityKalman2D:
    """2D constant-velocity Kalman filter with state [x, y, vx, vy]."""

    def __init__(
        self,
        x0: float,
        y0: float,
        t0: float,
        pos_sigma: float = 2.0,
        vel_sigma: float = 50.0,
        meas_sigma: float = 1.5,
    ):
        self.state = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([pos_sigma**2, pos_sigma**2, vel_sigma**2, vel_sigma**2])
        self.R = np.diag([meas_sigma**2, meas_sigma**2])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self.last_t = t0
        self.q_accel = 25.0

    def predict(self, t: float):
        dt = max(0.0, t - self.last_t)
        if dt == 0.0:
            return
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        q_pos = (dt**3) / 3.0 * (self.q_accel**2)
        q_vel = dt * (self.q_accel**2)
        q_pv = 0.5 * (dt**2) * (self.q_accel**2)
        Q = np.array([
            [q_pos, 0.0, q_pv, 0.0],
            [0.0, q_pos, 0.0, q_pv],
            [q_pv, 0.0, q_vel, 0.0],
            [0.0, q_pv, 0.0, q_vel],
        ], dtype=np.float64)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.last_t = t

    def update(self, z_x: float, z_y: float, gate_radius: float = 25.0) -> bool:
        z = np.array([z_x, z_y], dtype=np.float64)
        innov = z - self.state[:2]
        if np.linalg.norm(innov) > gate_radius:
            return False

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innov
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return True


def load_ground_truth(gt_path: str):
    t_list, x_list, y_list = [], [], []
    with open(gt_path, "r") as f:
        for row in csv.DictReader(f):
            t_list.append(float(row["timestamp_s"]))
            x_list.append(float(row["x"]))
            y_list.append(float(row["y"]))
    return np.array(t_list), np.array(x_list), np.array(y_list)


def extract_cluster_centroid(slice_ev: np.ndarray, prev_xy=None, search_radius: float = 35.0):
    if len(slice_ev) < 5:
        return None, 0

    xs = slice_ev[:, 1].astype(np.float64)
    ys = slice_ev[:, 2].astype(np.float64)

    if prev_xy is not None:
        d = np.hypot(xs - prev_xy[0], ys - prev_xy[1])
        mask = d <= search_radius
        if np.sum(mask) >= 5:
            xs = xs[mask]
            ys = ys[mask]

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    counts = np.sum((dx * dx + dy * dy) <= (8.0**2), axis=1)
    best_idx = int(np.argmax(counts))
    if counts[best_idx] < 4:
        return None, 0

    cx, cy = xs[best_idx], ys[best_idx]
    near = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (8.0**2)
    return (float(np.mean(xs[near])), float(np.mean(ys[near]))), int(np.sum(near))


def run_tracker(h5_path: str, gt_path: str, dt_step: float = 0.005):
    with h5py.File(h5_path, "r") as f:
        ev = f["events"][:]

    t_ev = ev[:, 0].astype(np.float64) * 1e-6
    order = np.argsort(t_ev)
    ev = ev[order]
    t_ev = t_ev[order]

    gt_t, gt_x, gt_y = load_ground_truth(gt_path)

    t_start = time.perf_counter()
    t_min, t_max = float(t_ev[0]), float(t_ev[-1])
    bins = np.arange(t_min, t_max + dt_step, dt_step)

    kf = None
    est_t, est_x, est_y = [], [], []
    updates = 0
    total_steps = 0

    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i + 1]
        t_mid = 0.5 * (b0 + b1)
        idx0 = np.searchsorted(t_ev, b0, side="left")
        idx1 = np.searchsorted(t_ev, b1, side="left")
        slice_ev = ev[idx0:idx1]
        total_steps += 1

        if kf is None:
            centroid, _ = extract_cluster_centroid(slice_ev)
            if centroid is not None:
                kf = ConstantVelocityKalman2D(centroid[0], centroid[1], t_mid)
                est_t.append(t_mid)
                est_x.append(kf.state[0])
                est_y.append(kf.state[1])
                updates += 1
        else:
            kf.predict(t_mid)
            centroid, _ = extract_cluster_centroid(slice_ev, prev_xy=kf.state[:2], search_radius=35.0)
            if centroid is not None and kf.update(centroid[0], centroid[1], gate_radius=25.0):
                updates += 1
            est_t.append(t_mid)
            est_x.append(kf.state[0])
            est_y.append(kf.state[1])

    elapsed_s = time.perf_counter() - t_start
    est_t = np.array(est_t)
    est_x = np.array(est_x)
    est_y = np.array(est_y)

    ref_x = np.interp(est_t, gt_t, gt_x)
    ref_y = np.interp(est_t, gt_t, gt_y)
    err_px = np.hypot(est_x - ref_x, est_y - ref_y)

    print("=== Classical Baseline: Event Clustering + 2D Kalman Filter ===")
    print(f"Time slice step     : {dt_step * 1000:.1f} ms ({total_steps} steps over {t_max - t_min:.3f}s)")
    print(f"Successful updates  : {updates} / {total_steps} ({100.0 * updates / max(1, total_steps):.1f}%)")
    print(f"Position RMSE       : {np.sqrt(np.mean(err_px**2)):.2f} px")
    print(f"Median / Max Error  : {np.median(err_px):.2f} px / {np.max(err_px):.2f} px")
    print(f"Total runtime       : {elapsed_s * 1000:.1f} ms ({(elapsed_s / max(1, total_steps)) * 1000:.3f} ms/step)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(gt_x, gt_y, "w--", lw=1.5, label="Ground Truth")
    ax.plot(est_x, est_y, "-", c="#38bdf8", lw=2.0, label="Kalman Estimate")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 346)
    ax.set_ylim(260, 0)
    ax.set_title("Trajectory Tracking: Kalman vs. Ground Truth")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.plot(est_t, err_px, c="#f59e0b", lw=1.8)
    ax.set_title(f"Localization Error (RMSE = {np.sqrt(np.mean(err_px**2)):.2f} px)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (px)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "docs/kalman_baseline_eval.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved evaluation plot to {plot_path}")


if __name__ == "__main__":
    run_tracker(
        "data/synth_iss_test/events.h5",
        "data/synth_iss_test/ground_truth.csv",
        dt_step=0.005,
    )

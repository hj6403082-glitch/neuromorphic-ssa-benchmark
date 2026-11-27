import csv
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

h5_path = "data/synth_iss_test/events.h5"
gt_path = "data/synth_iss_test/ground_truth.csv"
out_path = "docs/synth_iss_events.png"

with h5py.File(h5_path, "r") as f:
    # v2e HDF5 columns: [timestamp_us, x, y, polarity]
    ev = f["events"][:]

t_s = ev[:, 0].astype(np.float64) * 1e-6
x = ev[:, 1]
y = ev[:, 2]
p = ev[:, 3]

gt_t, gt_x, gt_y = [], [], []
with open(gt_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt_t.append(float(row["timestamp_s"]))
        gt_x.append(float(row["x"]))
        gt_y.append(float(row["y"]))
gt_t, gt_x, gt_y = np.array(gt_t), np.array(gt_x), np.array(gt_y)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Panel 1: Full 0.5s accumulated event map + ground truth track
ax = axes[0]
on = p == 1
off = p == 0
ax.scatter(x[off], y[off], s=0.4, c="#ef4444", alpha=0.4, label="OFF events")
ax.scatter(x[on], y[on], s=0.4, c="#22c55e", alpha=0.4, label="ON events")
ax.plot(gt_x, gt_y, c="white", lw=1.2, ls="--", label="Ground truth")
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 346)
ax.set_ylim(260, 0)
ax.set_title(f"Synthetic ISS Pass (0.5s, {len(ev):,} events)")
ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.legend(loc="lower right", markerscale=8, fontsize=8)

# Panel 2: Zoomed 50ms slice (0.20s - 0.25s) to inspect leading ON / trailing OFF polarity
ax = axes[1]
mask = (t_s >= 0.20) & (t_s <= 0.25)
gt_mask = (gt_t >= 0.20) & (gt_t <= 0.25)
ax.scatter(x[mask & off], y[mask & off], s=3, c="#ef4444", alpha=0.7, label="OFF")
ax.scatter(x[mask & on], y[mask & on], s=3, c="#22c55e", alpha=0.7, label="ON")
ax.plot(gt_x[gt_mask], gt_y[gt_mask], c="white", lw=1.5, ls="--", label="GT")
ax.set_facecolor("#0f172a")
ax.set_xlim(gt_x[gt_mask].min() - 15, gt_x[gt_mask].max() + 15)
ax.set_ylim(gt_y[gt_mask].max() + 15, gt_y[gt_mask].min() - 15)
ax.set_title("Zoomed 50 ms Window (0.20s–0.25s)")
ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved visualization to {out_path}")
print(f"Event time span: {t_s.min():.4f}s to {t_s.max():.4f}s | ON: {on.sum():,} | OFF: {off.sum():,}")

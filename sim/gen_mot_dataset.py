import argparse
import csv
import os
import h5py
import numpy as np


def generate_mot_pass(
    rng: np.random.Generator,
    duration_s: float = 2.0,
    width: int = 346,
    height: int = 260,
    n_targets: int = 3,
):
    """Generate a synthetic multi-object event stream with 2-4 low-SNR RSOs + heavy clutter."""
    dt = 0.005
    n_bins = int(round(duration_s / dt))
    t_all, x_all, y_all, p_all = [], [], [], []
    gt_rows = []

    # 1. Background shot noise (high clutter to match ISS_2_td density: ~80-180 ev/bin)
    bg_rate = rng.uniform(16000.0, 36000.0)
    n_bg = rng.poisson(bg_rate * duration_s)
    t_all.append(rng.uniform(0.0, duration_s, size=n_bg))
    x_all.append(rng.uniform(0.0, width - 1, size=n_bg))
    y_all.append(rng.uniform(0.0, height - 1, size=n_bg))
    p_all.append(rng.integers(0, 2, size=n_bg))

    # 2. Drifting background stars / slew clutter (5-12 static or slow-drifting distractors)
    n_stars = int(rng.integers(5, 13))
    slew_vx = rng.uniform(-15.0, 15.0)
    slew_vy = rng.uniform(-15.0, 15.0)
    for _ in range(n_stars):
        sx0 = rng.uniform(10.0, width - 10.0)
        sy0 = rng.uniform(10.0, height - 10.0)
        star_rate = rng.uniform(100.0, 600.0)
        n_s = rng.poisson(star_rate * duration_s)
        ts = rng.uniform(0.0, duration_s, size=n_s)
        xs = np.clip(sx0 + slew_vx * ts + rng.normal(0.0, 1.2, size=n_s), 0, width - 1)
        ys = np.clip(sy0 + slew_vy * ts + rng.normal(0.0, 1.2, size=n_s), 0, height - 1)
        t_all.append(ts)
        x_all.append(xs)
        y_all.append(ys)
        p_all.append(rng.integers(0, 2, size=n_s))

    # 3. 2-4 Moving RSO targets (varying SNR: 1.5 to 8 events/bin)
    for tid in range(n_targets):
        x0 = rng.uniform(30.0, width - 30.0)
        y0 = rng.uniform(30.0, height - 30.0)
        vx = rng.uniform(-65.0, 65.0)
        vy = rng.uniform(-55.0, 55.0)
        # Ensure target actually moves
        if np.hypot(vx, vy) < 12.0:
            vx = 25.0 if vx >= 0 else -25.0

        # Target event rate: 300 - 1600 ev/s (1.5 - 8.0 ev per 5ms bin)
        ev_rate = rng.uniform(300.0, 1600.0)
        n_ev = rng.poisson(ev_rate * duration_s)
        ts = rng.uniform(0.0, duration_s, size=n_ev)
        tx = x0 + vx * ts + rng.normal(0.0, 1.5, size=n_ev)
        ty = y0 + vy * ts + rng.normal(0.0, 1.5, size=n_ev)

        in_bounds = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        t_all.append(ts[in_bounds])
        x_all.append(tx[in_bounds])
        y_all.append(ty[in_bounds])
        p_all.append(rng.integers(0, 2, size=int(np.sum(in_bounds))))

        # Ground truth sampled at every 5ms bin center
        for b in range(n_bins):
            tb = (b + 0.5) * dt
            gx = x0 + vx * tb
            gy = y0 + vy * tb
            if 0.0 <= gx < width and 0.0 <= gy < height:
                gt_rows.append((tb, tid, gx, gy))

    t_cat = np.concatenate(t_all)
    x_cat = np.concatenate(x_all)
    y_cat = np.concatenate(y_all)
    p_cat = np.concatenate(p_all)
    order = np.argsort(t_cat)
    return t_cat[order], x_cat[order], y_cat[order], p_cat[order], gt_rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-object SSA dataset.")
    parser.add_argument("--out_root", default="data/benchmark_mot")
    parser.add_argument("--num_train", type=int, default=30)
    parser.add_argument("--num_val", type=int, default=6)
    parser.add_argument("--num_test", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_rows = []

    splits = [("train", args.num_train), ("val", args.num_val), ("test", args.num_test)]
    for split, count in splits:
        split_dir = os.path.join(args.out_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(count):
            n_targets = int(rng.integers(2, 5))
            pass_id = f"{split}_{i:03d}_mot_{n_targets}obj"
            h5_path = os.path.join(split_dir, f"{pass_id}.h5")
            gt_path = os.path.join(split_dir, f"{pass_id}_gt.csv")

            t, x, y, p, gt_rows = generate_mot_pass(rng, n_targets=n_targets)
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("t", data=t.astype(np.float64))
                f.create_dataset("x", data=x.astype(np.float32))
                f.create_dataset("y", data=y.astype(np.float32))
                f.create_dataset("p", data=p.astype(np.int8))

            with open(gt_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "id", "x", "y"])
                w.writerows(gt_rows)

            manifest_rows.append({
                "pass_id": pass_id,
                "split": split,
                "n_targets": n_targets,
                "events_h5": h5_path,
                "ground_truth_csv": gt_path,
            })

    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pass_id", "split", "n_targets", "events_h5", "ground_truth_csv"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} multi-object passes in {args.out_root} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()

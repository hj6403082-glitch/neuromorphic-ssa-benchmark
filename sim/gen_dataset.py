import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image


WIDTH, HEIGHT = 346, 260
V2E_SCRIPT = os.path.join(os.environ.get("V2E_DIR", os.path.expanduser("~/tools/v2e")), "v2e.py")


def render_starfield_frame(
    stars: np.ndarray,
    t: float,
    star_vx: float,
    star_vy: float,
    tx: float,
    ty: float,
    target_amp: float,
    target_sigma: float,
    bg_level: float = 12.0,
) -> np.ndarray:
    """Render a single 8-bit grayscale synthetic sky frame with drifting stars + RSO."""
    img = np.full((HEIGHT, WIDTH), bg_level, dtype=np.float32)

    sx = (stars[:, 0] + star_vx * t) % WIDTH
    sy = (stars[:, 1] + star_vy * t) % HEIGHT
    s_amp = stars[:, 2]

    for x0, y0, amp in zip(sx, sy, s_amp):
        ix, iy = int(round(x0)), int(round(y0))
        x_lo, x_hi = max(0, ix - 3), min(WIDTH, ix + 4)
        y_lo, y_hi = max(0, iy - 3), min(HEIGHT, iy + 4)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        psf = amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * 1.0**2))
        img[y_lo:y_hi, x_lo:x_hi] += psf

    rad = int(math.ceil(4.0 * target_sigma))
    ix, iy = int(round(tx)), int(round(ty))
    x_lo, x_hi = max(0, ix - rad), min(WIDTH, ix + rad + 1)
    y_lo, y_hi = max(0, iy - rad), min(HEIGHT, iy + rad + 1)
    if x_lo < x_hi and y_lo < y_hi:
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        psf = target_amp * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2.0 * target_sigma**2))
        img[y_lo:y_hi, x_lo:x_hi] += psf

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_pass(
    out_dir: str,
    seed: int,
    regime: str,
    duration_s: float = 0.4,
    fps: int = 1000,
):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "_tmp_frames")
    os.makedirs(frames_dir, exist_ok=True)

    if regime == "easy_staring":
        speed = rng.uniform(250.0, 450.0)
        target_amp = rng.uniform(160.0, 220.0)
        target_sigma = rng.uniform(1.6, 2.2)
        shot_hz = 0.05
        leak_hz = 0.05
        star_speed = rng.uniform(0.5, 2.0)
    elif regime == "medium_clutter":
        speed = rng.uniform(80.0, 250.0)
        target_amp = rng.uniform(100.0, 170.0)
        target_sigma = rng.uniform(1.4, 2.0)
        shot_hz = 0.15
        leak_hz = 0.15
        star_speed = rng.uniform(3.0, 12.0)
    else:  # hard_ebssa
        speed = rng.uniform(15.0, 80.0)
        target_amp = rng.uniform(60.0, 120.0)
        target_sigma = rng.uniform(1.3, 1.9)
        shot_hz = 0.30
        leak_hz = 0.25
        star_speed = rng.uniform(10.0, 30.0)

    angle = rng.uniform(0, 2.0 * math.pi)
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    cx = rng.uniform(100.0, WIDTH - 100.0)
    cy = rng.uniform(75.0, HEIGHT - 75.0)
    x0 = float(np.clip(cx - 0.5 * vx * duration_s, 20.0, WIDTH - 20.0))
    y0 = float(np.clip(cy - 0.5 * vy * duration_s, 20.0, HEIGHT - 20.0))
    if not (20.0 <= x0 + vx * duration_s <= WIDTH - 20.0):
        vx = -vx
    if not (20.0 <= y0 + vy * duration_s <= HEIGHT - 20.0):
        vy = -vy

    n_stars = int(rng.integers(25, 60))
    stars = np.column_stack([
        rng.uniform(0, WIDTH, n_stars),
        rng.uniform(0, HEIGHT, n_stars),
        rng.uniform(35.0, 140.0, n_stars),
    ])
    star_ang = rng.uniform(0, 2.0 * math.pi)
    star_vx = star_speed * math.cos(star_ang)
    star_vy = star_speed * math.sin(star_ang)

    n_frames = int(round(duration_s * fps))
    gt_path = os.path.join(out_dir, "ground_truth.csv")

    with open(gt_path, "w", newline="") as f_gt:
        writer = csv.writer(f_gt)
        writer.writerow(["timestamp_s", "x", "y", "vx", "vy"])
        for i in range(n_frames):
            t = i / float(fps)
            tx = x0 + vx * t
            ty = y0 + vy * t
            writer.writerow([f"{t:.6f}", f"{tx:.3f}", f"{ty:.3f}", f"{vx:.3f}", f"{vy:.3f}"])
            frame = render_starfield_frame(
                stars, t, star_vx, star_vy, tx, ty, target_amp, target_sigma
            )
            Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:06d}.png"))

    cmd = [
        sys.executable,
        V2E_SCRIPT,
        "-i", frames_dir,
        "--input_frame_rate", str(fps),
        "--dvs_exposure", "duration", "0.005",
        "--output_folder", out_dir,
        "--overwrite",
        "--dvs_h5", "events.h5",
        "--dvs_aedat2", "None",
        "--dvs_text", "None",
        "--no_preview",
        "--skip_video_output",
        "--disable_slomo",
        "--output_width", str(WIDTH),
        "--output_height", str(HEIGHT),
        "--pos_thres", "0.15",
        "--neg_thres", "0.28",
        "--sigma_thres", "0.03",
        "--cutoff_hz", "200",
        "--shot_noise_rate_hz", f"{shot_hz:.3f}",
        "--leak_rate_hz", f"{leak_hz:.3f}",
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shutil.rmtree(frames_dir, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(f"v2e failed for {out_dir}:\n{result.stderr.decode(errors='ignore')}")

    return {
        "regime": regime,
        "speed_px_s": round(speed, 2),
        "target_amp": round(target_amp, 1),
        "shot_hz": shot_hz,
        "leak_hz": leak_hz,
        "duration_s": duration_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate calibrated multi-pass SSA event dataset.")
    parser.add_argument("--out_root", default="data/benchmark_v1")
    parser.add_argument("--num_train", type=int, default=6)
    parser.add_argument("--num_val", type=int, default=2)
    parser.add_argument("--num_test", type=int, default=2)
    parser.add_argument("--duration_s", type=float, default=0.4)
    parser.add_argument("--fps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    regimes = ["easy_staring", "medium_clutter", "hard_ebssa"]
    splits = (
        [("train", i) for i in range(args.num_train)]
        + [("val", i) for i in range(args.num_val)]
        + [("test", i) for i in range(args.num_test)]
    )

    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as f_man:
        fields = [
            "pass_id", "split", "regime", "speed_px_s",
            "target_amp", "shot_hz", "leak_hz", "duration_s",
            "events_h5", "ground_truth_csv",
        ]
        writer = csv.DictWriter(f_man, fieldnames=fields)
        writer.writeheader()

        for idx, (split, s_idx) in enumerate(splits):
            regime = regimes[idx % len(regimes)]
            pass_id = f"{split}_{s_idx:03d}_{regime}"
            pass_dir = os.path.join(args.out_root, split, pass_id)
            print(f"[{idx + 1}/{len(splits)}] Generating {pass_id} ...", flush=True)

            meta = generate_pass(
                out_dir=pass_dir,
                seed=1000 + idx,
                regime=regime,
                duration_s=args.duration_s,
                fps=args.fps,
            )
            writer.writerow({
                "pass_id": pass_id,
                "split": split,
                **meta,
                "events_h5": os.path.join(pass_dir, "events.h5"),
                "ground_truth_csv": os.path.join(pass_dir, "ground_truth.csv"),
            })

    print(f"\nDone! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

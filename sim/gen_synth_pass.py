import csv
import os
import shutil
import struct
import subprocess
import cv2
import h5py
import numpy as np

repo_dir = os.path.abspath(".")
out_dir = os.path.join(repo_dir, "data", "synth_iss_test")
v2e_dir = os.path.expanduser("~/tools/v2e")
os.makedirs(out_dir, exist_ok=True)

W, H = 346, 260
FPS = 5000.0
DURATION_S = 0.5
N_FRAMES = int(FPS * DURATION_S)

# Target kinematics from your Chennai ISS pass (446.5 px/s at 6.4 arcsec/px)
rng = np.random.default_rng(42)
tx0, ty0 = 55.0, 75.0
vx, vy = 400.0, 199.0
target_sigma = 1.2
target_amp = 180.0

# Background sky + 45 synthetic stars with sidereal drift (~2.3 px/s)
bg_level = 5.0
starfield = np.full((H, W), bg_level, dtype=np.float32)
yy, xx = np.mgrid[0:H, 0:W]
for _ in range(45):
    sx = rng.uniform(5, W - 5)
    sy = rng.uniform(5, H - 5)
    s_amp = rng.uniform(15.0, 110.0)
    s_sig = rng.uniform(0.7, 1.3)
    starfield += s_amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * s_sig ** 2))

star_drift_vx = 2.3

uv_bin = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
cmd = [
    uv_bin, "run", "v2e",
    "--pipeframes",
    "--dvs346",
    "--dvs_h5", "events.h5",
    "--overwrite",
    "--no_preview",
    "--output_folder", out_dir,
]

print(f"Streaming {N_FRAMES} frames ({DURATION_S}s @ {int(FPS)} fps) into v2e...")
proc = subprocess.Popen(cmd, cwd=v2e_dir, stdin=subprocess.PIPE)

gt_path = os.path.join(out_dir, "ground_truth.csv")
with open(gt_path, "w", newline="") as gt_file:
    writer = csv.writer(gt_file)
    writer.writerow(["timestamp_s", "x", "y"])

    for i in range(N_FRAMES):
        t_s = i / FPS
        shift_x = star_drift_vx * t_s
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        frame = cv2.warpAffine(starfield, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_level)

        tx = tx0 + vx * t_s
        ty = ty0 + vy * t_s
        ix, iy = int(round(tx)), int(round(ty))
        r = 6
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        if x0 < x1 and y0 < y1:
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)[:, None]
            frame[y0:y1, x0:x1] += target_amp * np.exp(
                -((xs - tx) ** 2 + (ys - ty) ** 2) / (2.0 * target_sigma ** 2)
            )

        writer.writerow([f"{t_s:.6f}", f"{tx:.3f}", f"{ty:.3f}"])

        frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".bmp", frame_u8)
        if not ok:
            raise RuntimeError(f"Failed to encode frame {i}")
        raw = buf.tobytes()

        proc.stdin.write(struct.pack("<dI", t_s, len(raw)))
        proc.stdin.write(raw)
        proc.stdin.flush()

proc.stdin.close()
ret = proc.wait()
if ret != 0:
    raise RuntimeError(f"v2e exited with error code {ret}")

h5_path = os.path.join(out_dir, "events.h5")
print(f"\nSuccess! Ground truth saved to: {gt_path}")
print(f"Inspecting generated HDF5: {h5_path}")
with h5py.File(h5_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} | shape={obj.shape} | dtype={obj.dtype}")
        else:
            print(f"  Group  : {name}/")
    f.visititems(show)

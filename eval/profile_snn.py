import csv
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("models"))
sys.path.insert(0, os.path.abspath("eval"))

from snn_tracker import SpikingConvLIFTracker
from train_snn import load_pass_tensors, ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP
from eval_ebssa_zeroshot import load_ebssa_recording


def compute_layer_macs(h: int, w: int):
    """Theoretical dense MACs per 5 ms timestep for SpikingConvLIFTracker."""
    return {
        "conv1": 2 * 32 * 5 * 5 * h * w,
        "conv2": 32 * 32 * 3 * 3 * h * w,
        "conv3": 32 * 32 * 3 * 3 * h * w,
        "head": 32 * 1 * 1 * 1 * h * w,
    }


@torch.no_grad()
def measure_activity(model: SpikingConvLIFTracker, frames: torch.Tensor, device: torch.device):
    """Run sequence step-by-step and return per-layer input/spike sparsity."""
    x_seq = frames.unsqueeze(0).to(device)
    B, T, _, H, W = x_seq.shape

    mem1 = torch.zeros(B, model.conv1.out_channels, H, W, device=device)
    mem2 = torch.zeros(B, model.conv2.out_channels, H, W, device=device)
    mem3 = torch.zeros(B, model.conv3.out_channels, H, W, device=device)

    nz_in, s1_sum, s2_sum, s3_sum = 0.0, 0.0, 0.0, 0.0
    n_in, n_s1, n_s2, n_s3 = 0, 0, 0, 0

    for t in range(T):
        xt = x_seq[:, t]
        nz_in += (xt > 0).float().sum().item()
        n_in += xt.numel()

        cur1 = model.bn1(model.conv1(xt))
        spk1, mem1 = model.lif1(cur1, mem1)
        s1_sum += spk1.sum().item()
        n_s1 += spk1.numel()

        cur2 = model.bn2(model.conv2(spk1))
        spk2, mem2 = model.lif2(cur2, mem2)
        s2_sum += spk2.sum().item()
        n_s2 += spk2.numel()

        cur3 = model.bn3(model.conv3(spk2))
        spk3, mem3 = model.lif3(cur3, mem3)
        s3_sum += spk3.sum().item()
        n_s3 += spk3.numel()

    return {
        "in_nz": nz_in / max(1, n_in),
        "spk1": s1_sum / max(1, n_s1),
        "spk2": s2_sum / max(1, n_s2),
        "spk3": s3_sum / max(1, n_s3),
    }


@torch.no_grad()
def benchmark_latency(model: SpikingConvLIFTracker, device: torch.device, n_warmup: int = 50, n_steps: int = 500):
    """Measure single-step (5 ms bin) inference latency."""
    model = model.to(device).eval()
    x_step = torch.randn(1, 1, 2, GRID_H, GRID_W, device=device).clamp(min=0.0)

    # Warmup
    for _ in range(n_warmup):
        model(x_step)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        model(x_step)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / n_steps


def summarize_ops(name: str, act: dict, macs: dict):
    dense_total = sum(macs.values())
    # conv1 uses sparse event-count inputs (MACs); conv2/conv3/head use binary spikes (ACs)
    sparse_conv1_macs = macs["conv1"] * act["in_nz"]
    sparse_ac = (
        macs["conv2"] * act["spk1"]
        + macs["conv3"] * act["spk2"]
        + macs["head"] * act["spk3"]
    )
    sparse_total = sparse_conv1_macs + sparse_ac
    op_reduction = dense_total / max(1.0, sparse_total)

    # Horowitz (2014) 45 nm CMOS: 4.6 pJ per FP32 MAC, 0.9 pJ per FP32 AC
    dense_energy_uj = (dense_total * 4.6) * 1e-6
    snn_energy_uj = (sparse_conv1_macs * 4.6 + sparse_ac * 0.9) * 1e-6
    energy_ratio = dense_energy_uj / max(1e-9, snn_energy_uj)

    mean_spk = 100.0 * (act["spk1"] + act["spk2"] + act["spk3"]) / 3.0
    print(
        f"{name:<22} | {100*act['in_nz']:5.2f}% | {mean_spk:5.2f}% | "
        f"{dense_total/1e6:7.2f}M | {sparse_total/1e6:7.2f}M | "
        f"{op_reduction:5.1f}x | {snn_energy_uj:6.2f} uJ ({energy_ratio:4.1f}x less)"
    )


def main():
    ckpt = "checkpoints/snn_tracker_v2_best.pt"
    cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpikingConvLIFTracker(
        in_channels=2,
        hidden_channels=32,
        beta=0.85,
        orig_w=ORIG_W,
        orig_h=ORIG_H,
        grid_w=GRID_W,
        grid_h=GRID_H,
    ).to(cuda_dev)
    model.load_state_dict(torch.load(ckpt, map_location=cuda_dev, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== Model Footprint: SpikingConvLIFTracker (v2) ===")
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1024:.1f} KB FP32)")

    # 1. Latency benchmark
    print("\n=== Real-Time Inference Latency (Single 5 ms Event Bin) ===")
    if torch.cuda.is_available():
        gpu_ms = benchmark_latency(model, torch.device("cuda"))
        print(f"GPU ({torch.cuda.get_device_name(0)}): {gpu_ms:.3f} ms/step | {1000.0/gpu_ms:,.0f} steps/s ({5.0/gpu_ms:.1f}x real-time)")
    cpu_ms = benchmark_latency(model, torch.device("cpu"), n_warmup=10, n_steps=100)
    print(f"CPU: {cpu_ms:.3f} ms/step | {1000.0/cpu_ms:,.0f} steps/s ({5.0/cpu_ms:.1f}x real-time)")

    # Move back to GPU for activity profiling
    model = model.to(cuda_dev).eval()
    macs = compute_layer_macs(GRID_H, GRID_W)

    # 2. Sparsity & SynOps across benchmark_v2 test regimes + real-sky SL-16RB
    by_reg = {}
    with open("data/benchmark_v2/manifest.csv", "r") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            frames, _ = load_pass_tensors(row["events_h5"], row["ground_truth_csv"])
            act = measure_activity(model, frames, cuda_dev)
            by_reg.setdefault(row["regime"], []).append(act)

    # Real-sky SL-16RB
    ebssa = load_ebssa_recording("data/ebssa/labelled_ebssa.h5", "20170214-20-58_22285_SL-16RB", obj_id=None, duration_s=2.0)
    n_bins = int(round(ebssa["duration_s"] / DT_STEP))
    eb_frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)
    b_idx = np.floor(ebssa["t_ev"] / DT_STEP).astype(np.int64)
    valid = (b_idx >= 0) & (b_idx < n_bins)
    gx = np.clip((ebssa["x_ev"][valid] * (GRID_W / ebssa["x_max"])).astype(np.int64), 0, GRID_W - 1)
    gy = np.clip((ebssa["y_ev"][valid] * (GRID_H / ebssa["y_max"])).astype(np.int64), 0, GRID_H - 1)
    np.add.at(eb_frames, (b_idx[valid], ebssa["p_ev"][valid], gy, gx), 1.0)
    eb_act = measure_activity(model, torch.from_numpy(np.log1p(eb_frames)), cuda_dev)

    print("\n=== Neuromorphic Efficiency: Dense MACs vs. Sparse SynOps (per 5 ms Step) ===")
    print(f"{'Dataset / Regime':<22} | {'In NZ':>6} | {'Spk %':>6} | {'Dense':>8} | {'Sparse':>8} | {'OpRed':>6} | {'Est. Energy (45nm)':>20}")
    print("-" * 94)
    for reg, acts in by_reg.items():
        avg_act = {k: float(np.mean([a[k] for a in acts])) for k in acts[0]}
        summarize_ops(reg, avg_act, macs)
    summarize_ops("EBSSA SL-16RB (Real)", eb_act, macs)


if __name__ == "__main__":
    main()

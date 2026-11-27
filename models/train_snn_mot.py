import argparse, csv, os, sys, h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("models"))
from snn_tracker import SpikingConvLIFTracker

ORIG_W, ORIG_H, GRID_W, GRID_H, DT_STEP = 346, 260, 86, 65, 0.005


def load_mot_pass(h5_path: str, gt_csv: str, duration_s: float = 2.0, sigma: float = 1.5):
    n_bins = int(round(duration_s / DT_STEP))
    frames = np.zeros((n_bins, 2, GRID_H, GRID_W), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        t, x, y, p = f["t"][:], f["x"][:], f["y"][:], f["p"][:].astype(np.int64)
    b_idx = np.floor(t / DT_STEP).astype(np.int64)
    valid = (b_idx >= 0) & (b_idx < n_bins)
    gx = np.clip((x[valid] * (GRID_W / ORIG_W)).astype(np.int64), 0, GRID_W - 1)
    gy = np.clip((y[valid] * (GRID_H / ORIG_H)).astype(np.int64), 0, GRID_H - 1)
    np.add.at(frames, (b_idx[valid], p[valid], gy, gx), 1.0)
    frames = np.log1p(frames)

    heatmaps = np.zeros((n_bins, GRID_H, GRID_W), dtype=np.float32)
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    denom = 2.0 * (sigma ** 2)
    with open(gt_csv, "r") as f:
        for row in csv.DictReader(f):
            bi = int(np.floor(float(row["t"]) / DT_STEP))
            if 0 <= bi < n_bins:
                cx = int(np.clip(round(float(row["x"]) * (GRID_W / ORIG_W)), 0, GRID_W - 1))
                cy = int(np.clip(round(float(row["y"]) * (GRID_H / ORIG_H)), 0, GRID_H - 1))
                g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / denom)
                heatmaps[bi] = np.maximum(heatmaps[bi], g)
                heatmaps[bi, cy, cx] = 1.0
    return torch.from_numpy(frames), torch.from_numpy(heatmaps)


def centernet_focal_loss(logits, targets, alpha=2.0, beta=4.0):
    probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    pos_mask = targets.eq(1.0).float()
    neg_mask = targets.lt(1.0).float()
    pos_loss = -torch.log(probs) * torch.pow(1.0 - probs, alpha) * pos_mask
    neg_loss = -torch.log(1.0 - probs) * torch.pow(probs, alpha) * torch.pow(1.0 - targets, beta) * neg_mask
    return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp(min=1.0)


def forward_heatmap_logits(model, x_seq):
    B, T, _, H, W = x_seq.shape
    dev = x_seq.device
    m1 = torch.zeros(B, model.conv1.out_channels, H, W, device=dev)
    m2 = torch.zeros(B, model.conv2.out_channels, H, W, device=dev)
    m3 = torch.zeros(B, model.conv3.out_channels, H, W, device=dev)
    logits_list, spk_sum = [], 0.0
    for t in range(T):
        s1, m1 = model.lif1(model.bn1(model.conv1(x_seq[:, t])), m1)
        s2, m2 = model.lif2(model.bn2(model.conv2(s1)), m2)
        s3, m3 = model.lif3(model.bn3(model.conv3(s2)), m3)
        logits_list.append(model.head(s3 + 0.25 * m3).squeeze(1))
        spk_sum += (s1.mean() + s2.mean() + s3.mean()).item() / 3.0
    return torch.stack(logits_list, dim=1), spk_sum / max(1, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/benchmark_mot/manifest.csv")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--out_ckpt", default="checkpoints/snn_mot_best.pt")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    train_data, val_data = [], []
    with open(args.manifest, "r") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_data.append(load_mot_pass(row["events_h5"], row["ground_truth_csv"]))
            elif row["split"] == "val":
                val_data.append(load_mot_pass(row["events_h5"], row["ground_truth_csv"]))
    print(f"Loaded {len(train_data)} train / {len(val_data)} val MOT passes on {dev}.")

    model = SpikingConvLIFTracker(2, 32, args.beta, ORIG_W, ORIG_H, GRID_W, GRID_H).to(dev)
    nn.init.constant_(model.head.bias, -2.5)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val, rng = float("inf"), np.random.default_rng(42)

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_spk = 0.0
        for idx in rng.permutation(len(train_data)):
            x, y = train_data[idx][0].unsqueeze(0).to(dev), train_data[idx][1].unsqueeze(0).to(dev)
            if rng.random() < 0.5: x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
            if rng.random() < 0.5: x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
            opt.zero_grad()
            logits, spk = forward_heatmap_logits(model, x)
            loss = centernet_focal_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_loss += loss.item(); tr_spk += spk
        sched.step()
        tr_loss /= len(train_data); tr_spk /= len(train_data)

        model.eval()
        val_loss = val_spk = 0.0
        with torch.no_grad():
            for frames, hm in val_data:
                logits, spk = forward_heatmap_logits(model, frames.unsqueeze(0).to(dev))
                val_loss += centernet_focal_loss(logits, hm.unsqueeze(0).to(dev)).item()
                val_spk += spk
        val_loss /= len(val_data); val_spk /= len(val_data)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out_ckpt)
            saved = " *"
        if ep == 1 or ep % 5 == 0 or saved:
            print(f"Epoch {ep:02d}/{args.epochs} | Train: {tr_loss:.4f} ({100*tr_spk:4.1f}% spk) | Val: {val_loss:.4f} ({100*val_spk:4.1f}% spk){saved}")
    print(f"\nBest Val Focal Loss: {best_val:.4f} -> saved to {args.out_ckpt}")


if __name__ == "__main__":
    main()

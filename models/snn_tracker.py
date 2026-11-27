import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class SpikingConvLIFTracker(nn.Module):
    """Lightweight Spiking Conv-LIF tracker with differentiable 2D spatial soft-argmax."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        beta: float = 0.85,
        orig_w: int = 346,
        orig_h: int = 260,
        grid_w: int = 86,
        grid_h: int = 65,
    ):
        super().__init__()
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.grid_w = grid_w
        self.grid_h = grid_h

        spike_grad = surrogate.fast_sigmoid(slope=25.0)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Precompute coordinate grids in original pixel space
        xs = torch.linspace(0.0, orig_w - 1.0, grid_w)
        ys = torch.linspace(0.0, orig_h - 1.0, grid_h)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(1, -1))  # (1, H*W)
        self.register_buffer("grid_y", grid_y.reshape(1, -1))  # (1, H*W)

    def forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: (B, T, 2, grid_h, grid_w) binned ON/OFF event frames.
        Returns:
            coords: (B, T, 2) predicted (x, y) in original pixel coordinates.
            spike_rate: scalar mean spike rate across all LIF layers (sparsity metric).
        """
        B, T, _, H, W = x_seq.shape
        device = x_seq.device

        mem1 = torch.zeros(B, self.conv1.out_channels, H, W, device=device)
        mem2 = torch.zeros(B, self.conv2.out_channels, H, W, device=device)
        mem3 = torch.zeros(B, self.conv3.out_channels, H, W, device=device)

        coords_list = []
        total_spikes = 0.0
        total_neurons = 0

        for t in range(T):
            cur1 = self.bn1(self.conv1(x_seq[:, t]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            # Use membrane potential of final LIF + spikes for smooth spatial readout
            logits = self.head(spk3 + 0.25 * mem3).reshape(B, -1)
            probs = F.softmax(logits, dim=-1)

            pred_x = (probs * self.grid_x).sum(dim=-1)
            pred_y = (probs * self.grid_y).sum(dim=-1)
            coords_list.append(torch.stack([pred_x, pred_y], dim=-1))

            total_spikes = total_spikes + spk1.sum() + spk2.sum() + spk3.sum()
            total_neurons += spk1.numel() + spk2.numel() + spk3.numel()

        coords = torch.stack(coords_list, dim=1)  # (B, T, 2)
        spike_rate = total_spikes / max(1, total_neurons)
        return coords, spike_rate

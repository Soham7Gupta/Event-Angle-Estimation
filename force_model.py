# force_model.py
"""
Stage 2:  frozen angle model -> predicted angle (deg) -> ForceHead -> [CH3, CH4] (physical units)

The head works internally in a scaled (z-scored) force space so that derivatives are large,
and its output is mapped back to physical units by ForceScaler before any loss / metric.
"""

import torch
import torch.nn as nn

from config import Config
from model import EventWakeAngleModel


def angle_to_degrees(x, config):
    """Undo the [0, 1] target normalisation of the angle model."""
    if not config.normalize_angle_target:
        return x
    return x * (config.angle_max - config.angle_min) + config.angle_min


def load_frozen_angle_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    cfg.device = str(device)

    model = EventWakeAngleModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


class ForceScaler(nn.Module):
    """Per-channel affine map  physical <-> scaled.  Stored as buffers so it is saved
    with the head and inference needs no extra bookkeeping."""

    def __init__(self, n_channels):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_channels))
        self.register_buffer("std", torch.ones(n_channels))

    @torch.no_grad()
    def fit(self, targets):                      # targets: [R, C] physical units
        self.mean.copy_(targets.mean(dim=0))
        self.std.copy_(targets.std(dim=0).clamp_min(1e-12))

    def to_scaled(self, y):
        return (y - self.mean) / self.std

    def to_physical(self, z):
        return z * self.std + self.mean


class ForceHead(nn.Module):
    """angle (deg) -> forces. The MLP predicts SCALED forces; forward() returns PHYSICAL forces."""

    def __init__(self, n_outputs=2, hidden=32, angle_scale=25.0):
        super().__init__()
        self.angle_scale = angle_scale
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )
        self.scaler = ForceScaler(n_outputs)

    def forward_scaled(self, angle_deg):         # [B] -> [B, C] scaled
        return self.net((angle_deg / self.angle_scale).unsqueeze(-1))

    def forward(self, angle_deg):                # [B] -> [B, C] physical
        return self.scaler.to_physical(self.forward_scaled(angle_deg))


class EventForceModel(nn.Module):
    """Frozen angle model + trained force head (inference wrapper)."""

    def __init__(self, angle_model, force_head, config):
        super().__init__()
        self.angle_model = angle_model
        self.force_head = force_head
        self.config = config
        for p in self.angle_model.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.angle_model.eval()
        return self

    def forward(self, events, event_mask):
        with torch.no_grad():
            angle_norm = self.angle_model(events, event_mask)["angle"]
        angle_deg = angle_to_degrees(angle_norm, self.config)
        return {"angle_deg": angle_deg, "forces": self.force_head(angle_deg)}
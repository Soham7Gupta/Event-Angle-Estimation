# model.py
"""
Event-based angle regression model.

Each sample is a sequence of `sequence_length` temporal windows, each
holding up to `max_events_per_window` raw events. A small transformer
pools each window into an embedding, and a second transformer pools
those window embeddings across time into a single "wake latent" that
a small MLP head maps to a scalar angle.
"""

import torch
import torch.nn as nn


class EventEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    """Masked additive attention pooling over the sequence dimension.
    Rows that are fully masked (no valid entries) pool to zero rather
    than to NaN or an arbitrary uniform average."""

    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        # x: [B, N, D]   mask: [B, N] (True = valid)
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        has_valid = mask.any(dim=1, keepdim=True)
        return torch.where(has_valid, pooled, torch.zeros_like(pooled))


class EventWakeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.event_embed_dim

        self.event_embedding = EventEmbedding(D)
        self.event_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D, nhead=config.event_num_heads,
                dim_feedforward=config.event_ffn_dim, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=config.event_layers,
        )
        self.event_pool = AttentionPooling(D)

        self.temporal_position = nn.Parameter(torch.zeros(1, config.sequence_length, D))
        nn.init.normal_(self.temporal_position, std=0.02)

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D, nhead=config.temporal_num_heads,
                dim_feedforward=config.temporal_ffn_dim, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=config.temporal_layers,
        )
        self.temporal_pool = AttentionPooling(D)

        self.wake_projection = nn.Sequential(
            nn.Linear(D, config.wake_latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.wake_latent_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, events, event_mask):
        # events: [B, S, N, 4]   event_mask: [B, S, N]
        B, S, N, _ = events.shape

        flat_events = events.reshape(B * S, N, -1)
        flat_mask = event_mask.reshape(B * S, N)

        # A row that is fully masked (a temporal window with zero events)
        # would make every key in that row's self-attention -inf, and
        # softmax(-inf, ..., -inf) is NaN. Un-mask one (all-zero padding)
        # token for those rows so the transformer stays numerically safe;
        # `event_pool` below still zeroes such windows out using the
        # *real* mask, so this has no effect on the final representation.
        fully_empty = ~flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[fully_empty, 0] = True

        x = self.event_embedding(flat_events)
        x = self.event_transformer(x, src_key_padding_mask=~safe_mask)
        window_embeddings = self.event_pool(x, flat_mask).reshape(B, S, -1)

        window_embeddings = window_embeddings + self.temporal_position[:, :S]
        temporal = self.temporal_transformer(window_embeddings)

        step_has_events = event_mask.any(dim=2)   # [B, S]
        wake = self.temporal_pool(temporal, step_has_events)

        return self.wake_projection(wake)


class AngleHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.wake_latent_dim, config.angle_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.angle_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.angle_hidden_dim, 1),
        )

    def forward(self, wake):
        return self.net(wake).squeeze(-1)


class EventWakeAngleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wake_encoder = EventWakeEncoder(config)
        self.angle_head = AngleHead(config)

    def forward(self, events, event_mask):
        wake = self.wake_encoder(events, event_mask)
        return {"angle": self.angle_head(wake), "wake": wake}
# config.py
from __future__ import annotations

from dataclasses import dataclass

import torch

"""
Expected data layout (forces.xlsx, if present, is ignored — only the
.raw file and the folder/file name are used):

    data/
        0winkel/recording.raw
        5winkel/recording.raw
        -5winkel/recording.raw
        ...

Angle is parsed from the folder/file name. Supported patterns:
'0winkel', '22.5winkel', '-5winkel', 'angle_15', 'angle-15',
'angle 15', '15deg', '-5deg'.
"""


@dataclass
class Config:
    # --- data ---
    data_root: str = "data"
    min_recordings: int = 3

    # --- event stream ---
    event_time_scale: float = 1e-6   # Prophesee RAW timestamps are in microseconds
    delta_t: float = 0.01            # width of one temporal window (10 ms)
    sequence_length: int = 20        # windows per sample -> 200 ms of context
    target_stride: float = 0.01      # generate one training sample every 10 ms

    event_feature_dim: int = 4       # [x, y, relative_time, polarity]
    max_events_per_window: int = 512
    random_event_subsampling: bool = False
    max_empty_window_fraction: float = 0.5

    # --- model ---
    event_embed_dim: int = 64
    event_num_heads: int = 4
    event_layers: int = 2
    event_ffn_dim: int = 256
    temporal_num_heads: int = 4
    temporal_layers: int = 2
    temporal_ffn_dim: int = 256
    wake_latent_dim: int = 128
    angle_hidden_dim: int = 64
    dropout: float = 0.10

    # --- angle target ---
    # Must cover the full range of angles present in your data,
    # including sign. Values outside [angle_min, angle_max] get
    # silently clipped during normalization, so this MUST match
    # your actual dataset's min/max angle.
    normalize_angle_target: bool = True
    angle_min: float = -25.0
    angle_max: float = 25.0

    # --- training ---
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    grad_clip_norm: float = 1.0
    huber_beta: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 7
    lr_min: float = 1e-6
    early_stopping_patience: int = 15

    # --- recording-level split ---
    # Never split individual samples randomly: samples from the same
    # recording are highly correlated in time.
    train_recordings: tuple[str, ...] = (
        "-25winkel", "-20winkel", "-18winkel", "-16winkel", "-14winkel",
        "-10winkel", "-5winkel", "0winkel", "5winkel", "10winkel",
        "14winkel", "15winkel", "16winkel", "18winkel", "20winkel",
        "25winkel", "-12winkel",
    )
    val_recordings: tuple[str, ...] = (
        "12winkel",
    )
    test_recordings: tuple[str, ...] = (
        "-15winkel",
    )

    # --- misc ---
    checkpoint_path: str = "best_angle_model.pt"
    results_path: str = "angle_results.csv"
    seed: int = 42
    num_workers: int = 0
    print_diagnostics: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.validate()

    @property
    def required_history(self) -> float:
        return self.sequence_length * self.delta_t

    def validate(self):
        if self.delta_t <= 0:
            raise ValueError("delta_t must be > 0")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.max_events_per_window <= 0:
            raise ValueError("max_events_per_window must be > 0")
        if self.event_feature_dim != 4:
            raise ValueError("event_feature_dim must be 4: [x, y, relative_time, polarity]")
        if self.target_stride <= 0:
            raise ValueError("target_stride must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.angle_max <= self.angle_min:
            raise ValueError("angle_max must be greater than angle_min")
        if not 0.0 <= self.max_empty_window_fraction <= 1.0:
            raise ValueError("max_empty_window_fraction must be in [0, 1]")
        if self.event_embed_dim % self.event_num_heads != 0:
            raise ValueError("event_embed_dim must be divisible by event_num_heads")
        if self.event_embed_dim % self.temporal_num_heads != 0:
            raise ValueError("event_embed_dim must be divisible by temporal_num_heads")

        train = set(self.train_recordings)
        val = set(self.val_recordings)
        test = set(self.test_recordings)
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            raise ValueError(f"Recording split overlap: {sorted(overlap)}")
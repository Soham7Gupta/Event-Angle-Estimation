# data.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from metavision_core.event_io.raw_reader import RawReader


# ================================================================
# Angle parsing
# ================================================================

_ANGLE_PATTERNS = [
    r"(-?\d+(?:\.\d+)?)\s*winkel",       # 0winkel, 22.5winkel
    r"angle[_\-\s](-?\d+(?:\.\d+)?)",    # angle_15, angle-15, angle 15
    r"(-?\d+(?:\.\d+)?)\s*deg",          # 15deg, -5deg
]


def parse_angle_from_text(text: str) -> float:
    for pattern in _ANGLE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError(
        f"Could not parse an angle from '{text}'. Expected a name like "
        "'0winkel', '22.5winkel', 'angle_15', or '15deg'."
    )


# ================================================================
# Discovery
# ================================================================

@dataclass
class RecordingMeta:
    recording_id: str
    raw_path: Path
    angle_deg: float


def discover_recordings(config) -> list:
    root = Path(config.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist or is not a directory: {root}")

    raw_paths = sorted(root.rglob("*.raw"))
    if len(raw_paths) < config.min_recordings:
        raise RuntimeError(
            f"Found only {len(raw_paths)} .raw file(s) under {root}, "
            f"need at least {config.min_recordings}."
        )

    recordings, seen_ids = [], set()
    for raw_path in raw_paths:
        rel_dir = raw_path.parent.relative_to(root)
        recording_id = raw_path.stem if str(rel_dir) == "." else rel_dir.as_posix()

        if recording_id in seen_ids:
            raise RuntimeError(f"Duplicate recording id: {recording_id}")
        seen_ids.add(recording_id)

        angle = parse_angle_from_text(f"{raw_path.parent.name} {raw_path.name}")
        recordings.append(RecordingMeta(recording_id, raw_path, angle))

    print(f"Discovered {len(recordings)} recording(s):")
    for r in recordings:
        print(f"  {r.recording_id:25s} angle={r.angle_deg:.3f} deg")

    return recordings


# ================================================================
# RAW event loading
# ================================================================

@dataclass
class Recording:
    recording_id: str
    angle_deg: float
    events: np.ndarray   # [N, 4] columns: x, y, time_seconds, polarity
    height: int
    width: int


def load_events_from_raw(raw_path: Path, config):
    reader = RawReader(str(raw_path))

    # NOTE: Metavision's RawReader.get_size() returns (height, width).
    # If normalized x/y coordinates ever end up outside [0, 1], this
    # ordering is the first thing to double-check for your SDK version.
    height, width = (int(v) for v in reader.get_size())

    chunks = []
    while not reader.is_done():
        ev = reader.load_n_events(1_000_000)
        if ev is None or len(ev) == 0:
            continue
        chunks.append(np.column_stack([
            ev["x"].astype(np.float32),
            ev["y"].astype(np.float32),
            ev["t"].astype(np.float64) * config.event_time_scale,
            ev["p"].astype(np.float32),
        ]))

    if not chunks:
        raise RuntimeError(f"No events found in {raw_path}")
    events = np.concatenate(chunks, axis=0)

    valid = (
        np.isfinite(events[:, 2])
        & np.isfinite(events[:, 0]) & (events[:, 0] >= 0) & (events[:, 0] < width)
        & np.isfinite(events[:, 1]) & (events[:, 1] >= 0) & (events[:, 1] < height)
    )
    events = events[valid]
    if len(events) == 0:
        raise RuntimeError(f"No valid events found in {raw_path}")

    events[:, 3] = np.clip(events[:, 3], 0.0, 1.0)  # guard stray polarity values

    order = np.argsort(events[:, 2], kind="stable")
    events = events[order]
    events[:, 2] -= events[0, 2]                    # time starts at 0
    return events.astype(np.float32), height, width


def load_all_recordings(config) -> list:
    metas = discover_recordings(config)

    recordings = []
    for meta in metas:
        events, height, width = load_events_from_raw(meta.raw_path, config)
        duration = float(events[-1, 2] - events[0, 2])
        print(f"{meta.recording_id}: {len(events):,} events, "
              f"{duration:.2f}s, angle={meta.angle_deg:.2f} deg")
        recordings.append(Recording(meta.recording_id, meta.angle_deg, events, height, width))

    return recordings


# ================================================================
# Recording-level split
# ================================================================

def split_recordings(recordings: list, config):
    by_id = {r.recording_id: r for r in recordings}

    requested = set(config.train_recordings) | set(config.val_recordings) | set(config.test_recordings)
    missing = requested - by_id.keys()
    if missing:
        raise RuntimeError(f"Requested recordings not found: {sorted(missing)}. "
                            f"Available: {sorted(by_id)}")

    unassigned = by_id.keys() - requested
    if unassigned:
        raise RuntimeError(f"Recordings not assigned to any split: {sorted(unassigned)}")

    train = [by_id[i] for i in config.train_recordings]
    val = [by_id[i] for i in config.val_recordings]
    test = [by_id[i] for i in config.test_recordings]

    print(f"Split -> train: {list(config.train_recordings)}, "
          f"val: {list(config.val_recordings)}, test: {list(config.test_recordings)}")

    return train, val, test


# ================================================================
# Dataset
# ================================================================

def _window_edges(start: float, config) -> np.ndarray:
    return start + np.arange(config.sequence_length + 1) * config.delta_t


class AngleDataset(Dataset):
    """
    Every sample is a fixed-size window of `sequence_length` temporal
    bins (each `delta_t` seconds wide) of past events, ending at a
    target time, paired with that recording's (constant) angle label.
    """

    def __init__(self, recordings: list, config, name: str = "dataset"):
        if not recordings:
            raise RuntimeError(f"{name}: received zero recordings.")

        self.recordings = list(recordings)
        self.config = config
        self.name = name
        self.samples = []   # (recording_index, target_time)

        for r_idx, rec in enumerate(self.recordings):
            self._index_recording(r_idx, rec)

        if not self.samples:
            raise RuntimeError(f"{name}: no usable samples were generated.")

    def _index_recording(self, r_idx, rec):
        cfg = self.config
        t0, t1 = float(rec.events[0, 2]), float(rec.events[-1, 2])
        first_target = t0 + cfg.required_history

        if first_target > t1:
            raise RuntimeError(
                f"Recording '{rec.recording_id}' ({t1 - t0:.2f}s) is shorter than "
                f"the required history ({cfg.required_history:.2f}s)."
            )

        n_steps = int(np.floor((t1 - first_target) / cfg.target_stride)) + 1
        target_times = first_target + np.arange(n_steps) * cfg.target_stride

        timestamps = rec.events[:, 2]
        kept = 0
        for t in target_times:
            edges = _window_edges(t - cfg.required_history, cfg)
            bounds = np.searchsorted(timestamps, edges, side="left")
            empty_fraction = float(np.mean(np.diff(bounds) == 0))
            if empty_fraction <= cfg.max_empty_window_fraction:
                self.samples.append((r_idx, float(t)))
                kept += 1

        if cfg.print_diagnostics:
            print(f"  [{self.name}] {rec.recording_id:20s} angle={rec.angle_deg:6.2f} "
                  f"usable={kept}/{n_steps}")

    def __len__(self):
        return len(self.samples)

    def _window_features(self, window, window_start, width, height, rng=None):
        cfg = self.config
        max_n = cfg.max_events_per_window

        if len(window) == 0:
            return (np.zeros((max_n, cfg.event_feature_dim), dtype=np.float32),
                    np.zeros(max_n, dtype=bool))

        if len(window) > max_n:
            if rng is not None:
                idx = np.sort(rng.choice(len(window), max_n, replace=False))
            else:
                idx = np.linspace(0, len(window) - 1, max_n).astype(np.int64)
            window = window[idx]

        x = window[:, 0] / max(width - 1, 1)
        y = window[:, 1] / max(height - 1, 1)
        rel_t = np.clip((window[:, 2] - window_start) / cfg.delta_t, 0.0, 1.0)
        polarity = window[:, 3] * 2.0 - 1.0   # {0, 1} -> {-1, +1}

        feats = np.stack([x, y, rel_t, polarity], axis=1).astype(np.float32)
        mask = np.ones(max_n, dtype=bool)
        n = len(feats)
        if n < max_n:
            feats = np.pad(feats, ((0, max_n - n), (0, 0)))
            mask[n:] = False
        return feats, mask

    def __getitem__(self, index):
        cfg = self.config
        r_idx, target_time = self.samples[index]
        rec = self.recordings[r_idx]

        start = target_time - cfg.required_history
        edges = _window_edges(start, cfg)
        bounds = np.searchsorted(rec.events[:, 2], edges, side="left")
        rng = np.random.default_rng(cfg.seed + index) if cfg.random_event_subsampling else None

        feats_seq, mask_seq = [], []
        for step in range(cfg.sequence_length):
            lo, hi = bounds[step], bounds[step + 1]
            feats, mask = self._window_features(
                rec.events[lo:hi], edges[step], rec.width, rec.height, rng
            )
            feats_seq.append(feats)
            mask_seq.append(mask)

        angle = rec.angle_deg
        if cfg.normalize_angle_target:
            target = np.clip((angle - cfg.angle_min) / (cfg.angle_max - cfg.angle_min), 0.0, 1.0)
        else:
            target = angle

        return {
            "events": torch.from_numpy(np.stack(feats_seq)).float(),
            "event_mask": torch.from_numpy(np.stack(mask_seq)).bool(),
            "angle": torch.tensor(target, dtype=torch.float32),
            "angle_deg": torch.tensor(angle, dtype=torch.float32),
            "time": torch.tensor(target_time, dtype=torch.float32),
            "recording_index": torch.tensor(r_idx, dtype=torch.long),
        }
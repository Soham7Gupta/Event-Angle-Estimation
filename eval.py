# eval_duration.py
"""
Evaluates prediction accuracy and inference speed on a single
recording, using only the first N seconds of it, for a range of
durations. Useful for understanding the accuracy/latency trade-off
if you ever want to run this closer to real time.

Usage:
    python eval_duration.py
    python eval_duration.py --checkpoint best_angle_model.pt --recording 12winkel
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import discover_recordings, load_events_from_raw, Recording, AngleDataset
from model import EventWakeAngleModel


DURATIONS_SEC = [0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10]


def denormalize(values, config):
    if not config.normalize_angle_target:
        return values
    return values * (config.angle_max - config.angle_min) + config.angle_min


def load_single_recording(recording_id: str, config) -> Recording:
    metas = discover_recordings(config)
    match = next((m for m in metas if m.recording_id == recording_id), None)
    if match is None:
        available = [m.recording_id for m in metas]
        raise RuntimeError(f"'{recording_id}' not found. Available: {available}")

    events, height, width = load_events_from_raw(match.raw_path, config)
    return Recording(match.recording_id, match.angle_deg, events, height, width)


def truncate_recording(rec: Recording, duration: float) -> Recording:
    keep = rec.events[:, 2] <= duration
    return Recording(rec.recording_id, rec.angle_deg, rec.events[keep], rec.height, rec.width)


@torch.no_grad()
def run_and_time(model, loader, device):
    preds = []
    t0 = time.perf_counter()
    for batch in loader:
        events = batch["events"].to(device)
        mask = batch["event_mask"].to(device)
        output = model(events, mask)
        preds.append(output["angle"].cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(preds), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_angle_model.pt")
    parser.add_argument("--recording", default="12winkel")
    args = parser.parse_args()

    config = Config()
    device = torch.device(config.device)
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EventWakeAngleModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    full_rec = load_single_recording(args.recording, config)
    true_angle = full_rec.angle_deg
    full_duration = float(full_rec.events[-1, 2])
    print(f"Loaded '{args.recording}' (true angle={true_angle:.1f} deg, "
          f"full duration={full_duration:.2f}s)")

    # Warm up so the first timed duration isn't skewed by one-time
    # lazy-init overhead (CUDA context, cuDNN algorithm selection, etc.)
    dummy_events = torch.zeros(1, config.sequence_length, config.max_events_per_window, 4, device=device)
    dummy_mask = torch.zeros(1, config.sequence_length, config.max_events_per_window, dtype=torch.bool, device=device)
    with torch.no_grad():
        model(dummy_events, dummy_mask)

    print(f"\n{'duration':>9s} {'n_samples':>10s} {'infer_s':>9s} "
          f"{'ms/sample':>10s} {'pred_mean':>10s} {'MAE':>7s}")

    for duration in DURATIONS_SEC:
        if duration > full_duration:
            print(f"{duration:9.2f}   skipped (recording is only {full_duration:.2f}s long)")
            continue
        if duration < config.required_history:
            print(f"{duration:9.2f}   skipped (< required history of {config.required_history:.2f}s)")
            continue

        truncated = truncate_recording(full_rec, duration)
        try:
            ds = AngleDataset([truncated], config, name=f"{duration}s")
        except RuntimeError as e:
            print(f"{duration:9.2f}   skipped ({e})")
            continue

        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

        pred_norm, elapsed = run_and_time(model, loader, device)
        pred_deg = denormalize(pred_norm, config)
        mae = float(np.mean(np.abs(pred_deg - true_angle)))
        ms_per_sample = (elapsed / len(pred_deg)) * 1000

        print(f"{duration:9.2f} {len(pred_deg):10d} {elapsed:9.4f} "
              f"{ms_per_sample:10.3f} {pred_deg.mean():10.2f} {mae:7.3f}")


if __name__ == "__main__":
    main()

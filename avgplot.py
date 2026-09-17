# eval_sliding_window_all_angles.py
"""
Sliding-window inference (window=0.6s, step=0.1s) run across EVERY
recording in the dataset (i.e. every angle), with no plotting.

For each recording:
    - Slide a 0.6s window across the full recording, moving forward
      0.1s at a time (same "forget the oldest 0.1s" scheme as before):

          window 1: [0.0s, 0.6s)
          window 2: [0.1s, 0.7s)
          window 3: [0.2s, 0.8s)
          ...

    - Get one prediction per window.
    - Compute the mean predicted angle across all windows in that
      recording, and the mean absolute error against the recording's
      true (ground-truth) angle.

Results for all recordings are then printed as a well-formatted table
and saved to CSV.

Usage:
    python eval_sliding_window_all_angles.py
    python eval_sliding_window_all_angles.py --checkpoint best_angle_model.pt
    python eval_sliding_window_all_angles.py --window 0.6 --step 0.1
"""

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import discover_recordings, load_events_from_raw, Recording, AngleDataset
from model import EventWakeAngleModel


def denormalize(values, config):
    if not config.normalize_angle_target:
        return values
    return values * (config.angle_max - config.angle_min) + config.angle_min


def load_full_recording(meta, config) -> Recording:
    events, height, width = load_events_from_raw(meta.raw_path, config)
    return Recording(meta.recording_id, meta.angle_deg, events, height, width)


def extract_window(rec: Recording, start: float, end: float) -> Recording:
    """Return a Recording containing only events in [start, end),
    with timestamps shifted so the window itself starts at t=0."""
    times = rec.events[:, 2]
    mask = (times >= start) & (times < end)
    windowed_events = rec.events[mask].copy()
    windowed_events[:, 2] -= start  # re-base to window-relative time
    return Recording(rec.recording_id, rec.angle_deg, windowed_events, rec.height, rec.width)


@torch.no_grad()
def predict_single(model, rec: Recording, config, device):
    """Run the model on a single (already-windowed) recording and
    return the (denormalized) predicted angle, or None if the window
    had no usable events for the dataset builder."""
    try:
        ds = AngleDataset([rec], config, name="sliding_window")
    except RuntimeError:
        return None

    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    preds = []
    for batch in loader:
        events = batch["events"].to(device)
        mask = batch["event_mask"].to(device)
        output = model(events, mask)
        preds.append(output["angle"].cpu().numpy())

    if not preds:
        return None

    pred_norm = np.concatenate(preds)
    pred_deg = denormalize(pred_norm, config)
    return float(pred_deg.mean())


def run_sliding_window(model, full_rec, full_duration, window, step, config, device):
    """Runs one full sliding-window sweep for a single recording.
    Returns a numpy array of per-window predicted angles (deg)."""
    predictions = []

    start = 0.0
    while start + window <= full_duration:
        end = start + window
        windowed_rec = extract_window(full_rec, start, end)
        pred = predict_single(model, windowed_rec, config, device)
        if pred is not None:
            predictions.append(pred)
        start += step

    return np.array(predictions)


def print_table(rows):
    """rows: list of dicts with keys recording_id, true_angle_deg,
    n_windows, mean_pred_deg, mae_deg. Prints a well-aligned table."""
    headers = ["Recording", "True Angle (deg)", "N Windows",
               "Mean Pred (deg)", "MAE (deg)"]
    col_widths = [
        max(len(headers[0]), max(len(str(r["recording_id"])) for r in rows)),
        len(headers[1]),
        len(headers[2]),
        len(headers[3]),
        len(headers[4]),
    ]

    def fmt_row(cells):
        return " | ".join(str(c).rjust(w) if i > 0 else str(c).ljust(w)
                           for i, (c, w) in enumerate(zip(cells, col_widths)))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row([
            r["recording_id"],
            f"{r['true_angle_deg']:.2f}",
            r["n_windows"],
            f"{r['mean_pred_deg']:.2f}",
            f"{r['mae_deg']:.3f}",
        ]))


def save_table_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["recording_id", "true_angle_deg", "n_windows",
                           "mean_pred_deg", "mae_deg"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved table to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_angle_model.pt")
    parser.add_argument("--window", type=float, default=0.6, help="window length in seconds")
    parser.add_argument("--step", type=float, default=0.1, help="slide step in seconds")
    parser.add_argument("--csv-out", default="sliding_window_all_angles_summary.csv")
    args = parser.parse_args()

    config = Config()
    device = torch.device(config.device)
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EventWakeAngleModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Warm-up (avoid first-call overhead skewing the first recording's timing)
    dummy_events = torch.zeros(1, config.sequence_length, config.max_events_per_window, 4, device=device)
    dummy_mask = torch.zeros(1, config.sequence_length, config.max_events_per_window, dtype=torch.bool, device=device)
    with torch.no_grad():
        model(dummy_events, dummy_mask)

    window = args.window
    step = args.step

    if window < config.required_history:
        print(f"Warning: window ({window:.2f}s) is shorter than the model's "
              f"required history ({config.required_history:.2f}s); predictions "
              f"may be unreliable or the dataset builder may skip windows.")

    metas = discover_recordings(config)
    print(f"Found {len(metas)} recordings in dataset.\n")

    rows = []
    for meta in metas:
        full_rec = load_full_recording(meta, config)
        true_angle = full_rec.angle_deg
        full_duration = float(full_rec.events[-1, 2])

        if window > full_duration:
            print(f"Skipping '{meta.recording_id}': window ({window:.2f}s) "
                  f"longer than recording ({full_duration:.2f}s).")
            continue

        predictions = run_sliding_window(
            model, full_rec, full_duration, window, step, config, device
        )

        if len(predictions) == 0:
            print(f"Skipping '{meta.recording_id}': no usable windows.")
            continue

        mean_pred = float(predictions.mean())
        mae = float(np.mean(np.abs(predictions - true_angle)))

        rows.append({
            "recording_id": meta.recording_id,
            "true_angle_deg": true_angle,
            "n_windows": len(predictions),
            "mean_pred_deg": mean_pred,
            "mae_deg": mae,
        })

        print(f"'{meta.recording_id}': true={true_angle:.2f} deg, "
              f"n_windows={len(predictions)}, mean_pred={mean_pred:.2f} deg, "
              f"MAE={mae:.3f} deg")

    if not rows:
        print("\nNo results to report.")
        return

    # sort by true angle for a cleaner table
    rows.sort(key=lambda r: r["true_angle_deg"])

    print(f"\n=== Summary (window={window:.2f}s, step={step:.2f}s) ===\n")
    print_table(rows)

    overall_mae = float(np.mean([r["mae_deg"] for r in rows]))
    print(f"\nOverall mean MAE across all recordings: {overall_mae:.3f} deg")

    save_table_csv(rows, args.csv_out)


if __name__ == "__main__":
    main()
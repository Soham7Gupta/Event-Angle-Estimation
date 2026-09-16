# eval_sliding_window_multi.py
"""
Sliding-window inference on a single recording, run for SEVERAL window
sizes and overlaid on one plot (different color per window size, drawn
from a sequential colormap for visual continuity).

For each window size in WINDOW_SIZES_SEC:
    - The first prediction is made as soon as `window` seconds of data
      has arrived.
    - The window then slides forward by `step` seconds at a time,
      always keeping the most recent `window` seconds of events and
      "forgetting" the oldest `step` seconds.

      e.g. for window=0.4, step=0.1:
      window 1: [0.0s, 0.4s)
      window 2: [0.1s, 0.5s)
      window 3: [0.2s, 0.6s)
      ...

All predictions (for every window size) are first written out to a CSV
file, then plotted as line-only graphs (no markers, thicker lines) on
the SAME axes, each window size in its own cividis-derived color, with
a horizontal baseline at 12 degrees (the ground-truth angle for the
'12winkel' recording). The plot uses the 'Solarize_Light2' style sheet.
The y-axis is zoomed to 10-14 degrees by default to highlight the
region around the baseline.

NOTE ON TIMESTAMPS: this script re-bases each window's event timestamps
so the window starts at t=0 (i.e. event_time - window_start). This
mirrors what truncate_recording()/AngleDataset() effectively did in
eval_duration.py (which always started at t=0). If your model/dataset
pipeline expects absolute (not window-relative) timestamps, remove the
time-shift in `extract_window()` below.

Usage:
    python eval_sliding_window_multi.py
    python eval_sliding_window_multi.py --checkpoint best_angle_model.pt --recording 12winkel
    python eval_sliding_window_multi.py --step 0.1 --baseline 12
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import discover_recordings, load_events_from_raw, Recording, AngleDataset
from model import EventWakeAngleModel


WINDOW_SIZES_SEC = [0.4, 0.6, 0.8, 1.0]


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
    """Runs one full sliding-window sweep for a single window size.
    Returns (window_ends, predictions) as numpy arrays."""
    if window < config.required_history:
        print(f"Warning: window ({window:.2f}s) is shorter than the model's "
              f"required history ({config.required_history:.2f}s); predictions "
              f"may be unreliable or the dataset builder may skip windows.")

    window_ends = []
    predictions = []

    start = 0.0
    print(f"\n--- window={window:.2f}s, step={step:.2f}s ---")
    print(f"{'window':>16s} {'pred_deg':>10s}")
    while start + window <= full_duration:
        end = start + window
        windowed_rec = extract_window(full_rec, start, end)
        pred = predict_single(model, windowed_rec, config, device)

        if pred is None:
            print(f"[{start:5.2f},{end:5.2f})s   skipped (no usable events)")
        else:
            print(f"[{start:5.2f},{end:5.2f})s {pred:10.2f}")
            window_ends.append(end)
            predictions.append(pred)

        start += step

    return np.array(window_ends), np.array(predictions)


def save_results_csv(results, recording_id, true_angle, csv_path):
    """Write all (window_size, window_end, predicted_angle) rows to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording_id", "true_angle_deg", "window_size_s",
                          "window_end_s", "predicted_angle_deg"])
        for window, (window_ends, predictions) in results.items():
            for end, pred in zip(window_ends, predictions):
                writer.writerow([recording_id, true_angle, window, end, pred])
    print(f"Saved results to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_angle_model.pt")
    parser.add_argument("--recording", default="12winkel")
    parser.add_argument("--step", type=float, default=0.1, help="slide step in seconds")
    parser.add_argument("--baseline", type=float, default=12.0, help="baseline angle (deg) to plot")
    parser.add_argument("--out", default="sliding_window_predictions_multi.png")
    parser.add_argument("--csv-out", default="sliding_window_predictions_multi.csv")
    parser.add_argument("--xmin", type=float, default=0.0, help="x-axis (time) min")
    parser.add_argument("--xmax", type=float, default=10.0, help="x-axis (time) max")
    parser.add_argument("--ymin", type=float, default=10.0, help="y-axis (angle) min")
    parser.add_argument("--ymax", type=float, default=14.0, help="y-axis (angle) max")
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

    # Warm-up (avoid first-call overhead skewing anything downstream)
    dummy_events = torch.zeros(1, config.sequence_length, config.max_events_per_window, 4, device=device)
    dummy_mask = torch.zeros(1, config.sequence_length, config.max_events_per_window, dtype=torch.bool, device=device)
    with torch.no_grad():
        model(dummy_events, dummy_mask)

    step = args.step

    # --- Run the sliding-window sweep for each window size ---
    results = {}  # window_size -> (window_ends, predictions)
    for window in WINDOW_SIZES_SEC:
        window_ends, predictions = run_sliding_window(
            model, full_rec, full_duration, window, step, config, device
        )
        results[window] = (window_ends, predictions)

    # --- Save results to CSV before plotting ---
    save_results_csv(results, args.recording, true_angle, args.csv_out)

    # --- Plot: all window sizes overlaid, line-only, cividis colormap ---
    plt.style.use("Solarize_Light2")

    cmap = plt.get_cmap("cividis")
    colors = cmap(np.linspace(0.0, 1.0, len(WINDOW_SIZES_SEC)))

    plt.figure(figsize=(10, 5))

    for window, color in zip(WINDOW_SIZES_SEC, colors):
        window_ends, predictions = results[window]
        if len(predictions) == 0:
            print(f"No predictions for window={window:.2f}s — skipping in plot.")
            continue
        plt.plot(window_ends, predictions, color=color, linewidth=2.5,
                  label=f"window={window:.1f}s")

    plt.axhline(args.baseline, color="red", linestyle="--", linewidth=1.5,
                label=f"Baseline ({args.baseline:.0f} deg)")

    plt.xlabel("Time (s, window end / most recent event received)")
    plt.ylabel("Predicted angle (deg)")
    plt.title(f"Sliding-window predictions — '{args.recording}' "
              f"(step={step:.2f}s, multiple window sizes)")
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
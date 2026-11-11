# predict_force.py
"""
Usage:  python predict_force.py path/to/recording.raw
Frozen angle model -> predicted angle -> force head -> [CH3, CH4] in physical units.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_events_from_raw, Recording, AngleDataset
from force_model import ForceHead, EventForceModel, load_frozen_angle_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_path", type=str)
    p.add_argument("--angle_ckpt", default="/home/gupta/Desktop/EB/angle prediction model/best_angle_model.pt")
    p.add_argument("--force_ckpt", default="/home/gupta/Desktop/EB/angle prediction model/force_head.pt")
    p.add_argument("--out", default="force_predictions.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    angle_model, cfg = load_frozen_angle_model(args.angle_ckpt, device)
    ck = torch.load(args.force_ckpt, map_location=device, weights_only=False)
    head = ForceHead(ck["n_outputs"], ck["hidden"], ck["angle_scale"]).to(device)
    head.load_state_dict(ck["state_dict"])        # includes the scaler buffers
    head.eval()
    model = EventForceModel(angle_model, head, cfg).to(device).eval()

    events, h, w = load_events_from_raw(Path(args.raw_path), cfg)
    rec = Recording(Path(args.raw_path).stem, 0.0, events, h, w)   # angle label unused
    loader = DataLoader(AngleDataset([rec], cfg, "inference"), batch_size=16,
                        shuffle=False, num_workers=0)

    angles, forces, times = [], [], []
    for batch in loader:
        out = model(batch["events"].to(device), batch["event_mask"].to(device))
        angles.append(out["angle_deg"].cpu().numpy())
        forces.append(out["forces"].cpu().numpy())
        times.append(batch["time"].numpy())
    angles, forces, times = np.concatenate(angles), np.concatenate(forces), np.concatenate(times)

    print(f"Predicted angle : {angles.mean():.2f} deg (std over time {angles.std():.2f})")
    for i, c in enumerate(ck["channels"]):
        print(f"Predicted {c}   : {forces[:, i].mean():+.6f} (std over time {forces[:, i].std():.6f})")

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time_s", "pred_angle_deg"] + [f"pred_{c}" for c in ck["channels"]])
        for t, a, fr in zip(times, angles, forces):
            wr.writerow([f"{t:.4f}", f"{a:.4f}"] + [f"{v:.6f}" for v in fr])
    print(f"Saved per-sample predictions to {args.out}")


if __name__ == "__main__":
    main()
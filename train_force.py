# train_force.py
"""
1. Runs the frozen angle model once per labelled recording, caches the predicted angles.
2. Leave-one-recording-out CV (sweeping the input-noise level) vs. a mean baseline and a
   polynomial baseline.
3. Trains the final force head on all 19 recordings and saves it.

Usage:  python train_force.py --angle_ckpt best_angle_model.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (discover_recordings, load_events_from_raw, parse_angle_from_text,
                  Recording, AngleDataset)
from force_model import ForceHead, angle_to_degrees, load_frozen_angle_model

CHANNELS = ["CH3", "CH4"]

# angle (deg) -> (CH3, CH4)
FORCES_BY_ANGLE = {
     25: (-0.370475, -0.218783),
     18: (-0.369351, -0.218396),
     12: (-0.371270, -0.218011),
    -20: (-0.366388, -0.222344),
    -14: (-0.366421, -0.222788),
    -10: (-0.365940, -0.223441),
    -25: (-0.366199, -0.222834),
     20: (-0.369730, -0.218633),
    -15: (-0.366506, -0.222687),
    -18: (-0.366558, -0.222488),
     -5: (-0.367036, -0.222323),
    -16: (-0.366339, -0.222413),
      5: (-0.370375, -0.219701),
     14: (-0.371457, -0.217797),
     10: (-0.371210, -0.218645),
     16: (-0.369770, -0.218241),
     15: (-0.370231, -0.218278),
    -12: (-0.365537, -0.223692),
      0: (-0.368954, -0.221020),
}
FORCE_LABELS = {f"{a}winkel": v for a, v in FORCES_BY_ANGLE.items()}


# ================================================================
# Step 1: cache predicted angles from the frozen angle model
# ================================================================

@torch.no_grad()
def predict_angles_for_recording(model, cfg, rec, device, batch_size=16):
    ds = AngleDataset([rec], cfg, "force-features")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    out = []
    for batch in loader:
        o = model(batch["events"].to(device), batch["event_mask"].to(device))
        out.append(angle_to_degrees(o["angle"], cfg).cpu().numpy())
    return np.concatenate(out)


def build_or_load_cache(args, device):
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild_cache:
        d = np.load(cache)
        ids = [str(s) for s in d["rec_ids"]]
        if set(ids) == set(FORCE_LABELS):
            print(f"Loading cached angle predictions from {cache}")
            pred = np.split(d["pred_deg"], np.cumsum(d["lens"])[:-1])
            return ids, pred
        print("Cache does not match the current label set -> rebuilding.")

    model, cfg = load_frozen_angle_model(args.angle_ckpt, device)
    metas = discover_recordings(cfg)

    labelled = [m for m in metas if m.recording_id in FORCE_LABELS]
    not_found = sorted(set(FORCE_LABELS) - {m.recording_id for m in metas})
    unlabelled = [m.recording_id for m in metas if m.recording_id not in FORCE_LABELS]
    if not_found:
        print(f"WARNING: force labels without a recording: {not_found}")
    if unlabelled:
        print(f"No force label -> skipped: {unlabelled}")
    if len(labelled) < 4:
        raise RuntimeError("Too few labelled recordings.")

    ids, pred = [], []
    for m in labelled:                       # one recording at a time to save memory
        events, h, w = load_events_from_raw(m.raw_path, cfg)
        rec = Recording(m.recording_id, m.angle_deg, events, h, w)
        pred.append(predict_angles_for_recording(model, cfg, rec, device))
        ids.append(m.recording_id)
        del events, rec

    np.savez(cache, pred_deg=np.concatenate(pred), lens=np.array([len(p) for p in pred]),
             rec_ids=np.array(ids))
    print(f"Saved angle cache to {cache}")
    return ids, pred


class ForceData:
    def __init__(self, ids, pred_list):
        self.ids = ids
        self.pred = [np.asarray(p, dtype=np.float32) for p in pred_list]       # per-sample angles
        self.true_deg = np.array([parse_angle_from_text(i) for i in ids], dtype=np.float64)
        self.targets = np.array([FORCE_LABELS[i] for i in ids], dtype=np.float64)  # [R, C]
        self.mean_pred_deg = np.array([p.mean() for p in self.pred])


# ================================================================
# Step 2: force head training (CPU: the head is tiny)
# ================================================================

def fit_force_head(data, train_recs, noise_deg, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_recs = np.asarray(train_recs)
    n_rec, n_ch = len(train_recs), data.targets.shape[1]

    y_phys = torch.tensor(data.targets[train_recs], dtype=torch.float32)       # [Rt, C] physical
    true_deg = torch.tensor(data.true_deg[train_recs], dtype=torch.float32)    # [Rt]
    lens = torch.tensor([len(data.pred[r]) for r in train_recs], dtype=torch.long)
    pad = torch.zeros(n_rec, int(lens.max()))
    for i, r in enumerate(train_recs):
        pad[i, :len(data.pred[r])] = torch.from_numpy(data.pred[r])

    head = ForceHead(n_ch, args.hidden, args.angle_scale)
    head.scaler.fit(y_phys)                                  # scale factors from training recordings only
    loss_unit = head.scaler.std.detach().clone()             # constant, only keeps the loss O(1)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    B = args.batch_size
    head.train()
    for _ in range(args.steps):
        ridx = torch.randint(0, n_rec, (B,))                 # equal weight per recording
        j = (torch.rand(B) * lens[ridx]).long()
        j = torch.minimum(j, lens[ridx] - 1)
        x_pred = pad[ridx, j]                                # angle as the event model predicts it
        x_true = true_deg[ridx] + torch.randn(B) * noise_deg # true angle + simulated angle error
        x = torch.where(torch.rand(B) < args.pred_frac, x_pred, x_true)

        pred_phys = head(x)                                  # network output -> scaled back to physical units
        loss = loss_fn(pred_phys / loss_unit, y_phys[ridx] / loss_unit)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()

    head.eval()
    return head


@torch.no_grad()
def head_forces(head, angles):
    x = torch.as_tensor(np.atleast_1d(angles), dtype=torch.float32)
    return head(x).numpy()                                   # [N, C] physical units


# ================================================================
# Step 3: leave-one-recording-out CV
# ================================================================

def loo_cv(data, noise_deg, args):
    R, C = data.targets.shape
    out = {k: np.zeros((R, C)) for k in ("head", "oracle", "mean", "poly")}
    for held in range(R):
        tr = np.array([r for r in range(R) if r != held])
        head = fit_force_head(data, tr, noise_deg, args, args.seed)
        out["head"][held] = head_forces(head, data.pred[held]).mean(axis=0)        # event-model angle
        out["oracle"][held] = head_forces(head, data.true_deg[held]).mean(axis=0)  # true angle
        out["mean"][held] = data.targets[tr].mean(axis=0)
        for c in range(C):
            coef = np.polyfit(data.true_deg[tr] / 25.0, data.targets[tr, c], args.poly_degree)
            out["poly"][held, c] = np.polyval(coef, data.mean_pred_deg[held] / 25.0)
    return out


def mae(pred, data):
    return np.abs(pred - data.targets).mean(axis=0)


def nmae(pred, data):            # MAE in units of the target std, averaged over channels
    return float((mae(pred, data) / data.targets.std(axis=0)).mean())


def print_cv_report(data, out, args):
    print("\nPer-recording CV (recording-averaged prediction, input = event-model angle):")
    print(f"  {'recording':10s} {'ang_true':>8s} {'ang_pred':>8s} | "
          + " | ".join(f"{c} true / pred / err" for c in CHANNELS))
    for r in np.argsort(data.true_deg):
        cols = " | ".join(f"{data.targets[r, c]:+.5f} / {out['head'][r, c]:+.5f} / "
                          f"{out['head'][r, c] - data.targets[r, c]:+.5f}"
                          for c in range(len(CHANNELS)))
        print(f"  {data.ids[r]:10s} {data.true_deg[r]:8.2f} {data.mean_pred_deg[r]:8.2f} | {cols}")

    print("\nCV mean absolute error (physical units, lower is better):")
    for name, key in [("force head (event angle)", "head"),
                      ("force head (true angle)", "oracle"),
                      ("mean baseline", "mean"),
                      (f"poly deg {args.poly_degree} baseline", "poly")]:
        e = mae(out[key], data)
        print(f"  {name:28s} " + "  ".join(f"{c}: {e[i]:.6f}" for i, c in enumerate(CHANNELS))
              + f"   (normalised {nmae(out[key], data):.3f})")
    print("  target std over recordings   "
          + "  ".join(f"{c}: {data.targets[:, i].std():.6f}" for i, c in enumerate(CHANNELS)))


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angle_ckpt", default="best_angle_model.pt")
    p.add_argument("--cache", default="angle_pred_cache_v2.npz")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--force_ckpt", default="force_head.pt")
    p.add_argument("--cv_results", default="force_cv_results.csv")
    p.add_argument("--skip_cv", action="store_true", help="skip CV; use the first value of --noise_grid")
    p.add_argument("--noise_grid", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                   help="std (deg) of noise added to the TRUE angle during training; CV picks the best")
    p.add_argument("--pred_frac", type=float, default=0.5,
                   help="fraction of training inputs taken from the event model's predicted angles "
                        "(the rest are true angle + noise)")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--angle_scale", type=float, default=25.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--poly_degree", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Angle model device: {device} | force head trains on CPU")

    ids, pred_list = build_or_load_cache(args, device)
    data = ForceData(ids, pred_list)
    R = len(ids)
    print(f"\n{R} labelled recordings, "
          f"{sum(len(x) for x in data.pred)} cached angle predictions")
    print("Frozen angle model on these recordings "
          "(only -15winkel was never seen; 12winkel was the early-stopping set):")
    for r in np.argsort(data.true_deg):
        print(f"  {data.ids[r]:10s} true={data.true_deg[r]:7.2f}  pred_mean={data.mean_pred_deg[r]:7.2f}  "
              f"MAE={np.abs(data.pred[r] - data.true_deg[r]).mean():.3f} deg")

    # ---- CV + noise sweep ----
    if args.skip_cv:
        noise = args.noise_grid[0]
    else:
        results = {}
        print(f"\nLeave-one-recording-out CV over noise levels {args.noise_grid} ...")
        for nd in args.noise_grid:
            out = loo_cv(data, nd, args)
            results[nd] = out
            print(f"  noise {nd:4.2f} deg -> head normalised MAE {nmae(out['head'], data):.3f} "
                  f"(mean baseline {nmae(out['mean'], data):.3f})")
        noise = min(results, key=lambda k: nmae(results[k]["head"], data))
        print(f"\nBest noise level: {noise} deg")
        out = results[noise]
        print_cv_report(data, out, args)
        print("If the head is not clearly below the mean baseline, the forces are not predictable "
              "from angle at the noise level of this data.")

        with open(args.cv_results, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["recording_id", "true_angle", "mean_pred_angle"]
                        + [f"{c}_{k}" for c in CHANNELS for k in ("true", "pred", "err")])
            for r in range(R):
                row = [data.ids[r], f"{data.true_deg[r]:.3f}", f"{data.mean_pred_deg[r]:.3f}"]
                for c in range(len(CHANNELS)):
                    row += [f"{data.targets[r, c]:.6f}", f"{out['head'][r, c]:.6f}",
                            f"{out['head'][r, c] - data.targets[r, c]:+.6f}"]
                wr.writerow(row)
        print(f"Saved CV results to {args.cv_results}")

    # ---- final head on all recordings ----
    print(f"\nTraining final force head on all {R} recordings (noise {noise} deg)...")
    head = fit_force_head(data, np.arange(R), noise, args, args.seed)

    fit = np.array([head_forces(head, data.pred[r]).mean(axis=0) for r in range(R)])
    print("In-sample MAE: " + "  ".join(
        f"{c}: {np.abs(fit[:, i] - data.targets[:, i]).mean():.6f}" for i, c in enumerate(CHANNELS)))
    print("Learned curve (angle -> forces):")
    for a in range(-25, 26, 5):
        f = head_forces(head, float(a))[0]
        print(f"  {a:4d} deg  " + "  ".join(f"{c}={f[i]:+.6f}" for i, c in enumerate(CHANNELS)))

    torch.save({"state_dict": head.state_dict(), "n_outputs": len(CHANNELS),
                "hidden": args.hidden, "angle_scale": args.angle_scale,
                "channels": CHANNELS, "noise_deg": noise, "hparams": vars(args)},
               args.force_ckpt)
    print(f"Saved force head (incl. scaler) to {args.force_ckpt}")


if __name__ == "__main__":
    main()
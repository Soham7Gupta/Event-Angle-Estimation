# train.py
"""
Trains the event-based angle regression model with a strict
recording-level train/val/test split, early stopping on validation
loss, and a final per-recording test report.
"""

import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data import load_all_recordings, split_recordings, AngleDataset
from model import EventWakeAngleModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, loss_fn, optimizer=None, grad_clip=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_batches = 0.0, 0
    preds, targets, rec_idx, times = [], [], [], []

    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            events = batch["events"].to(device)
            mask = batch["event_mask"].to(device)
            angle = batch["angle"].to(device)

            output = model(events, mask)
            loss = loss_fn(output["angle"], angle)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds.append(output["angle"].detach().cpu().numpy())
            targets.append(angle.detach().cpu().numpy())
            rec_idx.append(batch["recording_index"].numpy())
            times.append(batch["time"].numpy())

    return {
        "loss": total_loss / max(n_batches, 1),
        "predictions": np.concatenate(preds),
        "targets": np.concatenate(targets),
        "recording_indices": np.concatenate(rec_idx),
        "times": np.concatenate(times),
    }


def mae_rmse(pred, true):
    err = pred - true
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def denormalize(values, config):
    if not config.normalize_angle_target:
        return values
    return values * (config.angle_max - config.angle_min) + config.angle_min


def main():
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device)
    print(f"Device: {device}")

    recordings = load_all_recordings(config)
    train_rec, val_rec, test_rec = split_recordings(recordings, config)

    train_ds = AngleDataset(train_rec, config, "train")
    val_ds = AngleDataset(val_rec, config, "val")
    test_ds = AngleDataset(test_rec, config, "test")
    print(f"Samples -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                          pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = EventWakeAngleModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor,
        patience=config.lr_patience, min_lr=config.lr_min,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nTraining...")
    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, loss_fn,
                                 optimizer=optimizer, grad_clip=config.grad_clip_norm)
        val_stats = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step(val_stats["loss"])

        val_mae, _ = mae_rmse(
            denormalize(val_stats["predictions"], config),
            denormalize(val_stats["targets"], config),
        )
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train {train_stats['loss']:.5f} | "
              f"val {val_stats['loss']:.5f} | val MAE {val_mae:.3f} deg | lr {lr:.2e}")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val_loss": best_val_loss, "config": vars(config)},
                       config.checkpoint_path)
            print("  saved new best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print("Early stopping.")
                break

    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    test_stats = run_epoch(model, test_loader, device, loss_fn)
    test_pred = denormalize(test_stats["predictions"], config)
    test_true = denormalize(test_stats["targets"], config)
    mae, rmse = mae_rmse(test_pred, test_true)
    bias = float(np.mean(test_pred - test_true))

    print(f"\nTest loss {test_stats['loss']:.6f} | MAE {mae:.4f} deg | "
          f"RMSE {rmse:.4f} deg | bias {bias:+.4f} deg")

    print("\nPer-recording test results:")
    for r_idx in np.unique(test_stats["recording_indices"]):
        m = test_stats["recording_indices"] == r_idx
        rec = test_rec[int(r_idx)]
        p, t = test_pred[m], test_true[m]
        r_mae, r_rmse = mae_rmse(p, t)
        print(f"  {rec.recording_id:20s} true={t[0]:6.2f} pred_mean={p.mean():6.2f} "
              f"MAE={r_mae:.3f} RMSE={r_rmse:.3f}")

    with open(config.results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording_id", "time_s", "true_angle_deg", "pred_angle_deg", "error_deg"])
        for i in range(len(test_pred)):
            rec = test_rec[int(test_stats["recording_indices"][i])]
            writer.writerow([rec.recording_id, f"{test_stats['times'][i]:.4f}",
                              f"{test_true[i]:.4f}", f"{test_pred[i]:.4f}",
                              f"{test_pred[i] - test_true[i]:+.4f}"])
    print(f"\nSaved per-sample test results to {config.results_path}")


if __name__ == "__main__":
    main()
"""Main training entrypoint.

    python -m training.train --config configs/config.yaml

Covers: AMP mixed precision, gradient accumulation, EMA, warmup+cosine LR,
periodic hard-negative pool refresh, and early stopping on val/mask_mAP.
See DESIGN.md section 8 for the full staged-training rationale (backbone
frozen warm-up -> joint fine-tune -> hard-negative mining phase).
"""
from __future__ import annotations

import argparse
import copy
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import SiameseDefectDataset, siamese_collate_fn
from datasets.pair_sampler import (
    SiamesePairBatchSampler, HardNegativePool, split_positive_negative,
)
from datasets.transforms import SiamesePairTransform
from models.siamese_instance_segmentation import SiameseInstanceSegmentation
from losses.total_loss import TotalLoss
from training.scheduler import WarmupCosineScheduler
from training.validate import validate


def build_dataloader(cfg: dict, split_file: str, train: bool) -> DataLoader:
    transform = SiamesePairTransform(tuple(cfg["data"]["image_size"]), train=train)
    dataset = SiameseDefectDataset(
        split_file, cfg["data"]["images_dir"], cfg["data"]["defect_classes"],
        transform=transform, train=train,
    )
    if train:
        groups = split_positive_negative(dataset)
        hard_pool = HardNegativePool(groups["negative"], cfg["data"]["pair_sampling"]["hard_negative_pool_size"])
        sampler = SiamesePairBatchSampler(
            groups["positive"], groups["negative"], hard_pool,
            batch_size=cfg["train"]["batch_size"],
            positive_ratio=cfg["data"]["pair_sampling"]["positive_ratio"],
            hard_negative_ratio=cfg["data"]["pair_sampling"]["hard_negative_ratio"],
            num_batches=max(1, len(dataset) // cfg["train"]["batch_size"]),
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=siamese_collate_fn,
                             num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"])
        return loader, dataset, hard_pool
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                         collate_fn=siamese_collate_fn, num_workers=cfg["data"]["num_workers"])
    return loader, dataset, None


def move_targets(targets, device):
    for t in targets:
        for k in ("boxes", "labels", "masks", "pair_label"):
            t[k] = t[k].to(device)
    return targets


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)

    model = SiameseInstanceSegmentation(cfg).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    criterion = TotalLoss(cfg).to(device)

    train_loader, train_dataset, hard_pool = build_dataloader(cfg, cfg["data"]["train_split"], train=True)
    val_loader, _, _ = build_dataloader(cfg, cfg["data"]["val_split"], train=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])
    accum_steps = cfg["experiment"]["grad_accum_steps"]
    optimizer_steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_steps = optimizer_steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = optimizer_steps_per_epoch * cfg["train"]["warmup_epochs"]
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps,
                                       cfg["train"]["min_lr_ratio"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["experiment"]["amp"])
    ema_decay = cfg["train"]["ema_decay"]

    best_metric, epochs_without_improve = -1.0, 0
    hn_cfg = cfg["train"]["hard_negative_mining"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        if hn_cfg["enabled"] and epoch >= hn_cfg["start_epoch"] and \
                epoch % cfg["data"]["pair_sampling"]["hard_negative_refresh_every_epochs"] == 0:
            hard_pool.refresh(model.encoder, train_dataset, device)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            reference = batch["reference"].to(device, non_blocking=True)
            inputs = batch["input"].to(device, non_blocking=True)
            targets = move_targets(batch["targets"], device)

            with torch.autocast(device_type=device.type, enabled=cfg["experiment"]["amp"]):
                outputs = model(reference, inputs)
                losses = criterion(outputs, targets)
                loss = losses["total"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items() if k != "total"} |
                              {"total": f"{losses['total'].item():.4f}",
                               "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        metrics = validate(ema_model, val_loader, device, cfg)
        print(f"[epoch {epoch}] val metrics: {metrics}")

        current = metrics[cfg["train"]["early_stopping"]["metric"].split("/")[-1]]
        if current > best_metric:
            best_metric = current
            epochs_without_improve = 0
            torch.save({"model": model.state_dict(), "ema_model": ema_model.state_dict(),
                        "epoch": epoch, "metric": best_metric},
                       os.path.join(cfg["experiment"]["output_dir"], "best.pt"))
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg["train"]["early_stopping"]["patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{epochs_without_improve} epochs)")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)

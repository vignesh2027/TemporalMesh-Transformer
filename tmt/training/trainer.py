"""
trainer.py — TMT training loop with wandb logging.

Trains on wikitext-2 (or tinystories) using AdamW + cosine warmup schedule.
Logs: train loss, val perplexity, exit rate per layer, and memory anchor norms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..model.config import TMTConfig
from ..model.model import TMTModel
from .loss import compute_loss
from .scheduler import cosine_warmup_scheduler


@dataclass
class TrainConfig:
    # Data
    dataset: str = "wikitext-2"         # or "tinystories"
    batch_size: int = 16
    seq_len: int = 256                  # shorter than max for memory efficiency

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500
    total_steps: int = 10_000

    # Saving
    save_dir: str = "checkpoints"
    save_every: int = 500
    eval_every: int = 100

    # Logging
    use_wandb: bool = False             # set True when wandb is configured
    project: str = "temporal-mesh-transformer"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss
    exit_gate_coeff: float = 0.1


class TMTTrainer:
    def __init__(
        self,
        model_cfg: TMTConfig,
        train_cfg: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)

        self.model = TMTModel(model_cfg).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step = 0

        if train_cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=train_cfg.project, config={
                    "model": vars(model_cfg),
                    "train": vars(train_cfg),
                })
                self.wandb = wandb
            except ImportError:
                print("wandb not installed — skipping wandb logging")
                self.wandb = None
        else:
            self.wandb = None

        os.makedirs(train_cfg.save_dir, exist_ok=True)
        print(self.model)

    def train(self) -> None:
        self.model.train()
        data_iter = iter(self.train_loader)

        while self.step < self.cfg.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            # Next-token prediction: targets are shifted by 1
            x = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward
            output = self.model(x)
            total_loss, ce_loss, gate_loss = compute_loss(
                output.logits,
                targets,
                output.confidences,
                self.cfg.exit_gate_coeff,
            )

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            self.step += 1

            # Logging
            if self.step % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                avg_exit_rate = self._compute_exit_rate(output)
                print(
                    f"step={self.step:5d} | loss={total_loss.item():.4f} | "
                    f"ce={ce_loss.item():.4f} | gate={gate_loss.item():.4f} | "
                    f"exit={avg_exit_rate:.3f} | lr={lr:.2e}"
                )
                if self.wandb:
                    self.wandb.log({
                        "loss/total": total_loss.item(),
                        "loss/ce": ce_loss.item(),
                        "loss/gate": gate_loss.item(),
                        "train/exit_rate": avg_exit_rate,
                        "train/lr": lr,
                        "step": self.step,
                    })

            if self.val_loader and self.step % self.cfg.eval_every == 0:
                val_ppl = self.evaluate()
                print(f"  val_perplexity={val_ppl:.2f}")
                if self.wandb:
                    self.wandb.log({"val/perplexity": val_ppl, "step": self.step})
                self.model.train()

            if self.step % self.cfg.save_every == 0:
                self._save(f"{self.cfg.save_dir}/ckpt_step{self.step}.pt")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            x, targets = input_ids[:, :-1], input_ids[:, 1:]
            out = self.model(x)
            loss, *_ = compute_loss(out.logits, targets, out.confidences)
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        import math
        return math.exp(avg_loss)

    @staticmethod
    def _compute_exit_rate(output) -> float:
        if not output.exit_masks:
            return 0.0
        final_mask = output.exit_masks[-1]
        return final_mask.float().mean().item()

    def _save(self, path: str) -> None:
        torch.save({
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
        print(f"  saved checkpoint → {path}")

"""
=============================================================
BanglaBERT Fine-Tuning — Bangla Fake News Classification
Baseline Model for Mamba Comparison
Optimized for: NVIDIA A100 40GB
=============================================================
Features:
  - BF16 mixed precision  (native A100 tensor cores)
  - Larger batch size     (32 × 2 accum = 64 effective)
  - Auto-resume from latest checkpoint if training interrupted
  - Saves checkpoint after every epoch (keeps latest 2)
  - Tracks & saves best model by Macro-F1 on validation set
  - Weighted loss for class imbalance (83.5% / 16.5%)
  - MLflow tracking (DagsHub) — one run covers train + eval
=============================================================
Expected runtime on A100-40GB:
  ~3 min/epoch  →  ~15 min total (5 epochs, BF16, batch=32)
=============================================================
PRE-REQUISITE:
  tokenized cache at Artifacts/tokenized_cache_bert/
  This class loads that cache directly — no tokenization here.

NOTE: Evaluation logic (Steps 7–10) lives in src/evaluate_bert.py.
============================================================="""

import os
import sys
import json
import glob
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_from_disk
import mlflow
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import save_json, load_json, section, create_directory
from src.evaluate_bert import BertEvaluate
from config.config import settings
from config.params import params


class BertFineTune:
    """
    BanglaBERT Fine-Tuning pipeline for Bangla Fake News Classification.

    Loads pre-tokenized HuggingFace datasets from disk, builds the model,
    trains with BF16 + OneCycleLR, saves checkpoints every epoch, and
    tracks the best val Macro-F1.

    A single MLflow run is opened in initialize_bert_finetuning() and spans
    both training (Steps 1–6) and evaluation (Steps 7–10, via BertEvaluate).
    DagsHub credentials are read from env vars:
      MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    """

    def __init__(self):
        # ── Paths (from config) ────────────────────────────────
        self.bert_cache_dir   = settings.bert_finetune.bert_cache_dir
        self.checkpoint_dir   = settings.bert_finetune.checkpoint_dir
        self.best_model_dir   = settings.bert_finetune.best_model_dir
        self.results_file     = settings.bert_finetune.results_file
        self.short_test_path  = settings.bert_finetune.short_test_subset
        self.long_test_path   = settings.bert_finetune.long_test_subset

        # ── Hyper-parameters (from params) ─────────────────────
        self.model_name    = params.bert.model_name
        self.num_labels    = params.bert.num_labels
        self.max_length    = params.bert.max_length
        self.epochs        = params.bert.epochs
        self.batch_size    = params.bert.batch_size
        self.grad_accum    = params.bert.grad_accum
        self.learning_rate = params.bert.learning_rate
        self.weight_decay  = params.bert.weight_decay
        self.warmup_pct    = params.bert.warmup_pct
        self.max_grad_norm = params.bert.max_grad_norm
        self.use_bf16      = params.bert.use_bf16
        self.class_weights = params.bert.class_weights
        self.num_workers   = params.bert.num_workers

        # Global setting
        self.seed          = settings.seed

        # ── Runtime state ──────────────────────────────────────
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer     = None
        self.model         = None
        self.optimizer     = None
        self.scheduler     = None
        self.criterion     = None
        self.train_loader  = None
        self.val_loader    = None
        self.test_loader   = None
        self.best_f1       = 0.0
        self.history       = []

    # ──────────────────────────────────────────────────────────
    # UTILITIES — Seed / device
    # ──────────────────────────────────────────────────────────
    def _set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    def _configure_device(self):
        """Log GPU info and disable BF16 if unsupported."""
        section("Device & Precision Setup")
        logging.info(f"  Device       : {self.device}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"  GPU          : {gpu_name}")
            logging.info(f"  VRAM         : {vram_gb:.1f} GB")
            logging.info(f"  BF16 support : {torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                logging.warning("  ⚠️  BF16 not supported — falling back to FP32")
                self.use_bf16 = False
        else:
            logging.warning("  ⚠️  No GPU found — running on CPU (will be very slow)")
            self.use_bf16 = False

        logging.info(f"  Precision    : {'BF16' if self.use_bf16 else 'FP32'}")
        logging.info(f"  Batch size   : {self.batch_size} × {self.grad_accum} accum "
                     f"= {self.batch_size * self.grad_accum} effective")
        logging.info(f"  Epochs       : {self.epochs}")
        logging.info(f"  LR           : {self.learning_rate}")

    # ──────────────────────────────────────────────────────────
    # CHECKPOINT UTILITIES
    # ──────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int):
        """
        Save full training state after each epoch.
        Keeps only the 2 most recent checkpoints to save disk space.
        BF16 mode: no scaler state to save.
        """
        try:
            create_directory(self.checkpoint_dir)
            path = self.checkpoint_dir / f"epoch_{epoch:02d}"
            create_directory(path)

            # Model weights + config (HuggingFace format)
            self.model.save_pretrained(path)

            # Optimizer, scheduler, metadata
            torch.save(
                {
                    "epoch"     : epoch,
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict(),
                    "best_f1"   : self.best_f1,
                    "history"   : self.history,
                },
                path / "training_state.pt",
            )
            logging.info(f"  💾 Checkpoint saved → {path}")

            # Keep only the latest 2 checkpoints
            all_ckpts = sorted(self.checkpoint_dir.glob("epoch_*"))
            for old in all_ckpts[:-2]:
                shutil.rmtree(old)
                logging.info(f"  🗑  Removed old checkpoint: {old}")

        except Exception as e:
            raise CustomException(e, sys)

    def _find_latest_checkpoint(self) -> Path | None:
        """Return path to the latest valid checkpoint, or None."""
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*")) \
            if self.checkpoint_dir.exists() else []
        for ckpt in reversed(ckpts):
            if (ckpt / "training_state.pt").exists() and \
               (ckpt / "config.json").exists():
                return ckpt
        return None

    def _load_checkpoint(self, ckpt_path: Path):
        """
        Restore optimizer + scheduler + metadata from checkpoint.
        Sets self.best_f1, self.history and returns last_completed_epoch.
        """
        try:
            state = torch.load(
                ckpt_path / "training_state.pt",
                map_location=self.device,
            )
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.best_f1 = state["best_f1"]
            self.history = state["history"]
        except Exception as e:
            raise CustomException(e, sys)

        logging.info(f"  ✅ Resumed from         : {ckpt_path}")
        logging.info(f"     Last completed epoch  : {state['epoch']}")
        logging.info(f"     Best val Macro-F1     : {state['best_f1']:.4f}")
        return state["epoch"]

    def _save_best_model(self, metrics: dict):
        """Save best model weights, tokenizer config, and its metrics."""
        try:
            create_directory(self.best_model_dir)
            self.model.save_pretrained(self.best_model_dir)
            self.tokenizer.save_pretrained(self.best_model_dir)
            with open(self.best_model_dir / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            raise CustomException(e, sys)

        logging.info(f"  🏆 Best model saved  → {self.best_model_dir}")
        logging.info(f"     Macro-F1 = {metrics['macro_f1']:.4f}  |  "
                     f"AUC-ROC = {metrics['auc_roc']:.4f}")

    # ──────────────────────────────────────────────────────────
    # STEP 1 — LOAD TOKENIZER
    # ──────────────────────────────────────────────────────────
    def load_tokenizer(self):
        """Load BanglaBERT tokenizer (only used for saving best model config)."""
        section("STEP 1 · Loading Tokenizer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise CustomException(e, sys)

        logging.info(f"  Loaded : {self.model_name}  (used for saving best model config)")
        logging.info(f"  Vocab  : {self.tokenizer.vocab_size:,}")

    # ──────────────────────────────────────────────────────────
    # STEP 2 — LOAD PRE-TOKENIZED DATASETS
    # ──────────────────────────────────────────────────────────
    def load_datasets(self):
        """
        Load the pre-tokenized HuggingFace datasets from disk.
        Requires bert_preprocess (OfflineTokenize with max_length=512) to
        have been run first.
        """
        section("STEP 2 · Loading Pre-Tokenized Datasets")
        for split in ["train", "val", "test"]:
            split_path = self.bert_cache_dir / split
            if not split_path.exists():
                raise CustomException(
                    FileNotFoundError(
                        f"Pre-tokenized cache not found at '{split_path}'.\n"
                        "Run OfflineTokenize (max_length=512) first to generate the cache."
                    ),
                    sys,
                )
        try:
            train_ds = load_from_disk(str(self.bert_cache_dir / "train"))
            val_ds   = load_from_disk(str(self.bert_cache_dir / "val"))
            test_ds  = load_from_disk(str(self.bert_cache_dir / "test"))
        except Exception as e:
            raise CustomException(e, sys)

        for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
            logging.info(f"  {name:5s}: {len(ds):,} rows")

        fmt_cols = ["input_ids", "attention_mask", "label"]
        train_ds.set_format(type="torch", columns=fmt_cols)
        val_ds.set_format(  type="torch", columns=fmt_cols)
        test_ds.set_format( type="torch", columns=fmt_cols)

        # ── DataLoaders ──────────────────────────────────────
        # pin_memory=True + num_workers maximises A100 GPU feed rate
        self.train_loader = DataLoader(
            train_ds,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size  = self.batch_size * 2,   # no backward → 2× batch is safe
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size  = self.batch_size * 2,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

        steps_per_epoch = (len(self.train_loader) + self.grad_accum - 1) // self.grad_accum
        total_steps     = steps_per_epoch * self.epochs
        logging.info(f"\n  Steps/epoch  : {steps_per_epoch}")
        logging.info(f"  Total steps  : {total_steps:,}")
        return total_steps

    # ──────────────────────────────────────────────────────────
    # STEP 3 — BUILD MODEL + OPTIMIZER + SCHEDULER
    # ──────────────────────────────────────────────────────────
    def build_model(self, total_steps: int):
        """
        Build or resume model, optimizer, loss, and OneCycleLR scheduler.
        Resumes from the latest checkpoint if one is found.
        """
        section("STEP 3 · Building Model")
        checkpoint_path = self._find_latest_checkpoint()

        try:
            if checkpoint_path:
                logging.info(f"  ♻️  Checkpoint found → {checkpoint_path}")
                logging.info("  Loading model weights from checkpoint...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(checkpoint_path),
                    num_labels              = self.num_labels,
                    ignore_mismatched_sizes = True,
                )
            else:
                logging.info(f"  No checkpoint — loading pretrained {self.model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels = self.num_labels,
                )
        except Exception as e:
            raise CustomException(e, sys)

        self.model = self.model.to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        logging.info(f"  Total params    : {total_params / 1e6:.1f}M")
        logging.info(f"  Trainable params: {trainable_params / 1e6:.1f}M")

        # ── Weighted loss ─────────────────────────────────────
        weight_tensor  = torch.tensor(
            self.class_weights, dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logging.info(f"  Loss weights → Fake={self.class_weights[0]:.4f}, "
                     f"Real={self.class_weights[1]:.4f}")

        # ── Optimizer ─────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
            eps          = 1e-8,
        )

        # ── OneCycleLR scheduler ──────────────────────────────
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr           = self.learning_rate,
            total_steps      = total_steps,
            pct_start        = self.warmup_pct,
            anneal_strategy  = "cos",
            div_factor       = 25,       # start LR = max_lr / 25
            final_div_factor = 1e4,      # end LR   = max_lr / (25 × 1e4)
        )

        start_epoch = 0
        if checkpoint_path:
            last_epoch  = self._load_checkpoint(checkpoint_path)
            start_epoch = last_epoch
            logging.info(f"  Next epoch to train : {start_epoch + 1}")
        else:
            logging.info("  Starting fresh training from epoch 1")

        return start_epoch

    # ──────────────────────────────────────────────────────────
    # STEP 4 — TRAIN ONE EPOCH
    # ──────────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch: int, reset_vram_peak: bool = False) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        steps      = 0

        # ── Reset VRAM peak counter before the training pass ────────────
        if reset_vram_peak and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            # BF16 forward pass — no GradScaler needed on A100
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.use_bf16):
                outputs = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                )
                loss = self.criterion(outputs.logits, labels)
                loss = loss / self.grad_accum

            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            preds       = outputs.logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            steps      += 1

            # Progress log every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = total_loss / steps
                acc      = correct / total * 100
                lr_now   = self.scheduler.get_last_lr()[0]
                logging.info(f"  Epoch {epoch} | Step {step+1:4d}/{len(self.train_loader)} | "
                             f"Loss={avg_loss:.4f} | Acc={acc:.1f}% | "
                             f"LR={lr_now:.2e}")

        # Handle leftover steps if dataset not divisible by grad_accum
        if len(self.train_loader) % self.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return total_loss / steps, correct / total * 100

    # ──────────────────────────────────────────────────────────
    # INTERNAL — Expected Calibration Error (ECE)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _compute_ece(
        labels: np.ndarray,
        probs: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error for binary classification.

        Bins samples by the predicted positive-class probability and
        measures the weighted average gap between mean confidence and
        observed accuracy in each bin.  Lower is better (0 = perfect
        calibration).

        Args:
            labels: Ground-truth binary labels (0 or 1).
            probs:  Predicted probability for the positive class (class 1).
            n_bins: Number of equal-width bins (default 10).

        Returns:
            ECE value in [0, 1].
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(labels)
        for i in range(n_bins):
            mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
            # First bin: include probs == 0.0
            if i == 0:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            avg_confidence = probs[mask].mean()
            avg_accuracy   = labels[mask].mean()
            ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)
        return ece

    # ──────────────────────────────────────────────────────────
    # STEP 5 — EVALUATE (validation only, during training loop)
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate_val(self, loader, split_name: str = "val") -> dict:
        """
        Lightweight validation evaluation used inside the training loop.
        Full test-set evaluation and thesis experiment are handled by
        BertEvaluate (src/evaluate_bert.py).
        """
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
            precision_score,
            recall_score,
            roc_auc_score,
            average_precision_score,
        )
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.use_bf16):
                outputs = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                )

            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        auc_roc  = roc_auc_score(all_labels, all_probs)
        p, r, f, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1]
        )
        report = classification_report(
            all_labels, all_preds,
            target_names=["Fake (0)", "Real (1)"],
            digits=4,
        )

        # ── New metrics ────────────────────────────────────────
        mcc          = matthews_corrcoef(all_labels, all_preds)
        pr_auc       = average_precision_score(all_labels, all_probs)
        weighted_p   = precision_score(all_labels, all_preds, average="weighted")
        weighted_r   = recall_score(all_labels, all_preds, average="weighted")
        weighted_f1  = f1_score(all_labels, all_preds, average="weighted")
        cm           = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        # ── Expected Calibration Error (ECE) ───────────────────
        ece = self._compute_ece(all_labels, all_probs, n_bins=10)

        metrics = {
            "split"              : split_name,
            "macro_f1"           : float(macro_f1),
            "auc_roc"            : float(auc_roc),
            "mcc"                : float(mcc),
            "pr_auc"             : float(pr_auc),
            "ece"                : float(ece),
            "weighted_precision" : float(weighted_p),
            "weighted_recall"    : float(weighted_r),
            "weighted_f1"        : float(weighted_f1),
            "confusion_matrix"   : cm.tolist(),
            "fake_precision"     : float(p[0]),
            "fake_recall"        : float(r[0]),
            "fake_f1"            : float(f[0]),
            "real_precision"     : float(p[1]),
            "real_recall"        : float(r[1]),
            "real_f1"            : float(f[1]),
        }

        logging.info(f"\n  [{split_name.upper()}]")
        logging.info(f"  Macro-F1 = {macro_f1:.4f}  |  AUC-ROC = {auc_roc:.4f}")
        logging.info(f"  MCC = {mcc:.4f}  |  PR-AUC = {pr_auc:.4f}  |  ECE = {ece:.4f}")
        logging.info(f"  Weighted →  P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}")
        logging.info(f"  Fake  →  P={p[0]:.4f}  R={r[0]:.4f}  F1={f[0]:.4f}")
        logging.info(f"  Real  →  P={p[1]:.4f}  R={r[1]:.4f}  F1={f[1]:.4f}")
        logging.info(f"  Confusion Matrix (rows=true, cols=pred):")
        logging.info(f"    Fake: {cm[0].tolist()}")
        logging.info(f"    Real: {cm[1].tolist()}")
        logging.info(f"\n{report}")

        return metrics

    # ──────────────────────────────────────────────────────────
    # STEP 6 — TRAINING LOOP
    # ──────────────────────────────────────────────────────────
    def train(self, start_epoch: int):
        section("STEP 6 · Training Loop")

        overall_peak_vram_mb = 0.0   # track across all epochs

        for epoch in range(start_epoch, self.epochs):
            logging.info(f"\n{'─'*55}")
            logging.info(f"  EPOCH {epoch + 1} / {self.epochs}")
            logging.info(f"{'─'*55}")

            t0 = time.time()
            # reset_vram_peak=True resets the CUDA peak counter before each epoch
            train_loss, train_acc = self._train_one_epoch(epoch + 1, reset_vram_peak=True)
            elapsed = time.time() - t0

            logging.info(f"\n  Epoch {epoch+1} finished in {elapsed/60:.1f} min")
            logging.info(f"  Train Loss = {train_loss:.4f}  |  "
                         f"Train Acc  = {train_acc:.1f}%")

            val_metrics = self._evaluate_val(self.val_loader, split_name="val")

            # ── Capture peak VRAM for this epoch (train + val combined) ────
            epoch_peak_vram_mb = 0.0
            if self.device == "cuda":
                epoch_peak_vram_mb = round(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 1
                )
                overall_peak_vram_mb = max(overall_peak_vram_mb, epoch_peak_vram_mb)
                logging.info(
                    f"  Peak VRAM (epoch {epoch+1}) : {epoch_peak_vram_mb:.0f} MB"
                )

            # Record history
            history_entry = {
                "epoch"      : epoch + 1,
                "time_min"   : round(elapsed / 60, 2),
                "train_loss" : round(train_loss, 4),
                "train_acc"  : round(train_acc,  2),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in val_metrics.items()},
            }
            if self.device == "cuda":
                history_entry["train_peak_vram_mb"] = epoch_peak_vram_mb
            self.history.append(history_entry)

            # ── MLflow: per-epoch train + val metrics ─────────
            if mlflow.active_run():
                epoch_metrics = {
                    "train/loss"   : round(train_loss, 4),
                    "train/acc"    : round(train_acc,  2),
                    "val/macro_f1" : round(val_metrics["macro_f1"], 4),
                    "val/auc_roc"  : round(val_metrics["auc_roc"],  4),
                    "val/fake_f1"  : round(val_metrics["fake_f1"],  4),
                    "val/real_f1"  : round(val_metrics["real_f1"],  4),
                    "val/mcc"      : round(val_metrics["mcc"], 4),
                    "val/pr_auc"   : round(val_metrics["pr_auc"], 4),
                    "val/ece"      : round(val_metrics["ece"], 4),
                }
                if self.device == "cuda":
                    epoch_metrics["train/peak_vram_mb"] = epoch_peak_vram_mb
                mlflow.log_metrics(epoch_metrics, step=epoch + 1)

            # Save best model if val Macro-F1 improved
            if val_metrics["macro_f1"] > self.best_f1:
                self.best_f1 = val_metrics["macro_f1"]
                self._save_best_model(val_metrics)
                logging.info(f"  🏆 New best Macro-F1 = {self.best_f1:.4f}")
            else:
                logging.info(f"  No improvement. Best = {self.best_f1:.4f}")

            # Save checkpoint (every epoch, keep latest 2)
            self._save_checkpoint(epoch + 1)

        # ── Log overall peak training VRAM once at end of training ──────
        if self.device == "cuda" and mlflow.active_run():
            mlflow.log_metric("train/overall_peak_vram_mb", overall_peak_vram_mb)
            logging.info(
                f"  Overall peak training VRAM : {overall_peak_vram_mb:.0f} MB"
            )

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full fine-tuning pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_bert_finetuning(self):
        """Execute the BanglaBERT fine-tuning pipeline (Steps 1–6).

        Opens a single MLflow run that spans both training and evaluation.
        After training completes, hands off to BertEvaluate for
        Steps 7–10 (test evaluation, thesis experiment, results & summary).

        DagsHub credentials are read from env vars:
          MLFLOW_TRACKING_USERNAME — DagsHub username
          MLFLOW_TRACKING_PASSWORD — DagsHub access token / password
        """
        section("BanglaBERT Fine-Tuning — A100 40GB")

        self._set_seed()
        self._configure_device()

        # ── MLflow experiment setup ────────────────────────────
        mlflow.set_tracking_uri(settings.bert_mlflow.tracking_uri)
        mlflow.set_experiment(settings.bert_mlflow.experiment_name)
        logging.info(f"  MLflow tracking URI : {settings.bert_mlflow.tracking_uri}")
        logging.info(f"  MLflow experiment   : {settings.bert_mlflow.experiment_name}")

        with mlflow.start_run(run_name=settings.bert_mlflow.run_name) as _run:
            logging.info(f"  MLflow run ID       : {_run.info.run_id}")

            # ── Log all hyper-parameters at run start ──────────
            mlflow.log_params({
                "model_name"        : self.model_name,
                "num_labels"        : self.num_labels,
                "max_length"        : self.max_length,
                "epochs"            : self.epochs,
                "batch_size"        : self.batch_size,
                "grad_accum"        : self.grad_accum,
                "effective_batch"   : self.batch_size * self.grad_accum,
                "learning_rate"     : self.learning_rate,
                "weight_decay"      : self.weight_decay,
                "warmup_pct"        : self.warmup_pct,
                "max_grad_norm"     : self.max_grad_norm,
                "use_bf16"          : self.use_bf16,
                "class_weight_fake" : self.class_weights[0],
                "class_weight_real" : self.class_weights[1],
                "seed"              : self.seed,
            })
            mlflow.set_tags({
                "device"    : self.device,
                "framework" : "huggingface-transformers",
                "task"      : "text-classification",
                "language"  : "bengali",
            })

            self.load_tokenizer()
            total_steps = self.load_datasets()
            start_epoch = self.build_model(total_steps)
            self.train(start_epoch)   # logs per-epoch metrics internally

            # ── Build training config snapshot for the evaluator ──
            training_config = {
                "model"          : self.model_name,
                "max_length"     : self.max_length,
                "epochs"         : self.epochs,
                "batch_size"     : self.batch_size,
                "grad_accum"     : self.grad_accum,
                "effective_batch": self.batch_size * self.grad_accum,
                "lr"             : self.learning_rate,
                "precision"      : "BF16" if self.use_bf16 else "FP32",
                "class_weights"  : self.class_weights,
            }

            # ── Delegate evaluation (Steps 7–10) to BertEvaluate ──
            # BertEvaluate runs inside this same active MLflow run;
            # it logs final test metrics, results JSON, and the best model.
            evaluator = BertEvaluate()
            evaluator.initialize_bert_evaluation(
                test_loader      = self.test_loader,
                tokenizer        = self.tokenizer,
                training_config  = training_config,
                training_history = self.history,
                best_val_f1      = self.best_f1,
            )
        # ── MLflow run closes automatically here ──────────────
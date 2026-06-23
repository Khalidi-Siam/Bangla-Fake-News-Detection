"""
=============================================================
ssm_train.py — Bangla-Mamba Training Pipeline (Steps 1–6)
Optimized for: NVIDIA A100 40GB
=============================================================
Features:
  - Loads pre-tokenized 768-token cache from preprocessing
  - BF16 mixed precision (A100 native tensor cores)
  - AdamW + OneCycleLR with warmup
  - Weighted CrossEntropyLoss for 83.5/16.5 imbalance
  - Auto-resume from latest checkpoint if interrupted
  - Saves checkpoint after every epoch (keeps latest 2)
  - Best model saved by Val Macro-F1
  - MLflow tracking (DagsHub) — one run spans train + eval
=============================================================
Expected runtime on A100-40GB:
  ~6–10 min/epoch  →  ~35–55 min total (5 epochs, BF16)
=============================================================
PRE-REQUISITE:
  tokenized cache at Artifacts/tokenized_cache_mamba/
  This class loads that cache directly — no tokenization here.

NOTE: Evaluation logic (Steps 7–10) lives in src/evaluate_ssm.py.
=============================================================
"""

import sys
import json
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
import mlflow
import warnings
warnings.filterwarnings("ignore")

# Import our model
from src.ssm_model import build_bangla_mamba, BanglaMambaForClassification
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import section, create_directory
from src.evaluate_ssm import MambaEvaluate
from config.config import settings
from config.params import params


class MambaTrainer:
    """
    Bangla-Mamba (SSM) training pipeline for Bangla Fake News Classification.

    Loads pre-tokenized HuggingFace datasets from disk, builds the Mamba model
    from scratch, trains with BF16 + OneCycleLR, saves checkpoints every epoch,
    and tracks the best val Macro-F1.

    A single MLflow run is opened in initialize_mamba_training() and spans
    both training (Steps 1–6) and evaluation (Steps 7–10, via MambaEvaluate).
    DagsHub credentials are read from env vars:
      MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD

    Paths are read from config.settings.mamba_train.
    Hyper-parameters are read from config.params.mamba.
    """

    def __init__(self):
        # ── Paths (from config) ────────────────────────────────
        self.cache_dir      = settings.mamba_train.cache_dir
        self.checkpoint_dir = settings.mamba_train.checkpoint_dir
        self.best_model_dir = settings.mamba_train.best_model_dir
        self.results_file   = settings.mamba_train.results_file

        # ── Hyper-parameters (from params) ─────────────────────
        self.tokenizer_name = params.mamba.tokenizer_name
        self.vocab_size     = params.mamba.vocab_size
        self.d_model        = params.mamba.d_model
        self.n_layer        = params.mamba.n_layer
        self.num_labels     = params.mamba.num_labels
        self.max_length     = params.mamba.max_length
        self.epochs         = params.mamba.epochs
        self.batch_size     = params.mamba.batch_size
        self.grad_accum     = params.mamba.grad_accum
        self.learning_rate  = params.mamba.learning_rate
        self.weight_decay   = params.mamba.weight_decay
        self.warmup_pct     = params.mamba.warmup_pct
        self.max_grad_norm  = params.mamba.max_grad_norm
        self.use_bf16       = params.mamba.use_bf16
        self.class_weights  = params.mamba.class_weights
        self.num_workers    = params.mamba.num_workers

        # Global setting
        self.seed = settings.seed

        # ── Runtime state ──────────────────────────────────────
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer    = None
        self.model        = None
        self.optimizer    = None
        self.scheduler    = None
        self.criterion    = None
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None
        self.best_f1      = 0.0
        self.history      = []

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
        logging.info(f"  Max length   : {self.max_length} tokens")
        logging.info(f"  Epochs       : {self.epochs}")
        logging.info(f"  LR           : {self.learning_rate}")

    # ──────────────────────────────────────────────────────────
    # CHECKPOINT UTILITIES
    # ──────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int):
        """
        Save full training state after each completed epoch.
        Keeps only the 2 most recent checkpoints to save disk space.
        """
        try:
            create_directory(self.checkpoint_dir)
            path = self.checkpoint_dir / f"epoch_{epoch:02d}"
            create_directory(path)

            # Model weights + mamba config
            self.model.save(str(path))

            # Optimizer + scheduler + metadata
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
            has_weights = (ckpt / "model_weights.pt").exists()
            has_config  = (ckpt / "mamba_config.json").exists()
            has_state   = (ckpt / "training_state.pt").exists()
            if has_weights and has_config and has_state:
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
        logging.info(f"     Last completed epoch  : {state['epoch'] + 1}")
        logging.info(f"     Best val Macro-F1     : {state['best_f1']:.4f}")
        return state["epoch"]

    def _save_best_model(self, metrics: dict):
        """Save best model weights + config + tokenizer + metrics."""
        try:
            create_directory(self.best_model_dir)
            self.model.save(str(self.best_model_dir))
            self.tokenizer.save_pretrained(str(self.best_model_dir))
            with open(self.best_model_dir / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            raise CustomException(e, sys)

        logging.info(f"  🏆 Best model & tokenizer saved  → {self.best_model_dir}")
        logging.info(f"     Macro-F1 = {metrics['macro_f1']:.4f}  |  "
                     f"AUC-ROC  = {metrics['auc_roc']:.4f}")

    # ──────────────────────────────────────────────────────────
    # STEP 1 — LOAD TOKENIZER
    # ──────────────────────────────────────────────────────────
    def load_tokenizer(self):
        """Load BanglaBERT tokenizer (vocab reference + subset eval)."""
        section("STEP 1 · Loading Tokenizer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:
            raise CustomException(e, sys)

        # Override vocab_size with actual tokenizer value
        self.vocab_size = self.tokenizer.vocab_size
        logging.info(f"  Loaded : {self.tokenizer_name}  (used for saving best model config)")
        logging.info(f"  Vocab  : {self.vocab_size:,}")

    # ──────────────────────────────────────────────────────────
    # STEP 2 — LOAD PRE-TOKENIZED DATASETS
    # ──────────────────────────────────────────────────────────
    def load_datasets(self) -> int:
        """
        Load the pre-tokenized HuggingFace datasets from disk.
        Requires OfflineTokenize (max_length=mamba.max_length) to have run first.
        Returns total_steps for the OneCycleLR scheduler.
        """
        section("STEP 2 · Loading Pre-Tokenized Datasets")
        for split in ["train", "val", "test"]:
            split_path = self.cache_dir / split
            if not split_path.exists():
                raise CustomException(
                    FileNotFoundError(
                        f"Pre-tokenized cache not found at '{split_path}'.\n"
                        "Run OfflineTokenize (max_length=768) first to generate the cache."
                    ),
                    sys,
                )

        try:
            train_ds = load_from_disk(str(self.cache_dir / "train"))
            val_ds   = load_from_disk(str(self.cache_dir / "val"))
            test_ds  = load_from_disk(str(self.cache_dir / "test"))
        except Exception as e:
            raise CustomException(e, sys)

        for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
            logging.info(f"  {name:5s}: {len(ds):,} rows")

        fmt_cols = ["input_ids", "attention_mask", "label"]
        train_ds.set_format(type="torch", columns=fmt_cols)
        val_ds.set_format(  type="torch", columns=fmt_cols)
        test_ds.set_format( type="torch", columns=fmt_cols)

        # ── DataLoaders ──────────────────────────────────────
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

        steps_per_epoch = len(self.train_loader) // self.grad_accum
        total_steps     = steps_per_epoch * self.epochs
        logging.info(f"\n  Steps/epoch  : {steps_per_epoch}")
        logging.info(f"  Total steps  : {total_steps:,}")
        return total_steps

    # ──────────────────────────────────────────────────────────
    # STEP 3 — BUILD MODEL + OPTIMIZER + SCHEDULER
    # ──────────────────────────────────────────────────────────
    def build_model(self, total_steps: int) -> int:
        """
        Build or resume model, optimizer, loss, and OneCycleLR scheduler.
        Resumes from the latest checkpoint if one is found.
        Returns start_epoch.
        """
        section("STEP 3 · Building Bangla-Mamba Model")
        checkpoint_path = self._find_latest_checkpoint()

        try:
            if checkpoint_path:
                logging.info(f"  ♻️  Checkpoint found → {checkpoint_path}")
                logging.info("  Loading Mamba weights from checkpoint...")
                self.model = BanglaMambaForClassification.load(
                    str(checkpoint_path), device=self.device
                )
            else:
                logging.info("  No checkpoint — building fresh Mamba model")
                self.model = build_bangla_mamba(
                    vocab_size=self.vocab_size,
                    d_model=self.d_model,
                    n_layer=self.n_layer,
                    num_labels=self.num_labels,
                )
                self.model = self.model.to(self.device)
        except Exception as e:
            raise CustomException(e, sys)

        model_params = self.model.count_parameters()
        logging.info(f"  Backbone params : {model_params['backbone_M']}M")
        logging.info(f"  Head params     : {model_params['head_M']}M")
        logging.info(f"  Total params    : {model_params['total_M']}M")

        # ── Weighted loss ─────────────────────────────────────
        weight_tensor  = torch.tensor(
            self.class_weights, dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logging.info(f"  Loss weights → Fake={self.class_weights[0]:.4f}  "
                     f"Real={self.class_weights[1]:.4f}")

        # ── Optimizer ─────────────────────────────────────────
        # Mamba trains from scratch → higher LR (1e-3) vs BERT fine-tune (2e-5)
        # betas=(0.9, 0.95) slightly better for SSMs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           = self.learning_rate,
            weight_decay = self.weight_decay,
            eps          = 1e-8,
            betas        = (0.9, 0.95),
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

        # ── Resume state if checkpoint exists ─────────────────
        start_epoch = 0
        if checkpoint_path:
            last_epoch  = self._load_checkpoint(checkpoint_path)
            start_epoch = last_epoch + 1
            logging.info(f"  Resuming from epoch {start_epoch + 1}")
        else:
            logging.info("  Starting fresh training from epoch 1")

        return start_epoch

    # ──────────────────────────────────────────────────────────
    # STEP 4 — TRAIN ONE EPOCH
    # ──────────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch: int) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        steps      = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            # BF16 forward — no GradScaler needed on A100
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=self.use_bf16):
                logits = self.model(input_ids, attention_mask)
                loss   = self.criterion(logits, labels)
                loss   = loss / self.grad_accum

            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            steps      += 1

            # Progress log every 50 steps
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / steps
                acc      = correct / total * 100
                lr_now   = self.scheduler.get_last_lr()[0]
                logging.info(
                    f"  Epoch {epoch} | Step {step+1:4d}/{len(self.train_loader)} | "
                    f"Loss={avg_loss:.4f} | Acc={acc:.1f}% | LR={lr_now:.2e}"
                )

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
    # STEP 5 — EVALUATE (validation only, during training loop)
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate_val(self, loader, split_name: str = "val") -> dict:
        """
        Lightweight validation evaluation used inside the training loop.
        Full test-set evaluation and thesis experiment are handled by
        MambaEvaluate (src/evaluate_ssm.py).
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"]

            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=self.use_bf16):
                logits = self.model(input_ids, attention_mask)

            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)

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

        metrics = {
            "split"         : split_name,
            "macro_f1"      : float(macro_f1),
            "auc_roc"       : float(auc_roc),
            "fake_precision": float(p[0]),
            "fake_recall"   : float(r[0]),
            "fake_f1"       : float(f[0]),
            "real_precision": float(p[1]),
            "real_recall"   : float(r[1]),
            "real_f1"       : float(f[1]),
        }

        logging.info(f"\n  [{split_name.upper()}]")
        logging.info(f"  Macro-F1 = {macro_f1:.4f}  |  AUC-ROC = {auc_roc:.4f}")
        logging.info(f"  Fake  →  P={p[0]:.4f}  R={r[0]:.4f}  F1={f[0]:.4f}")
        logging.info(f"  Real  →  P={p[1]:.4f}  R={r[1]:.4f}  F1={f[1]:.4f}")
        logging.info(f"\n{report}")

        return metrics

    # ──────────────────────────────────────────────────────────
    # STEP 6 — TRAINING LOOP
    # ──────────────────────────────────────────────────────────
    def train(self, start_epoch: int):
        section("STEP 6 · Training Loop")

        for epoch in range(start_epoch, self.epochs):
            logging.info(f"\n{'─'*55}")
            logging.info(f"  EPOCH {epoch + 1} / {self.epochs}")
            logging.info(f"{'─'*55}")

            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch + 1)
            elapsed = time.time() - t0

            logging.info(f"\n  Epoch {epoch+1} finished in {elapsed/60:.1f} min")
            logging.info(f"  Train Loss = {train_loss:.4f}  |  "
                         f"Train Acc  = {train_acc:.1f}%")

            val_metrics = self._evaluate_val(self.val_loader, split_name="val")

            # Record history
            self.history.append({
                "epoch"      : epoch + 1,
                "time_min"   : round(elapsed / 60, 2),
                "train_loss" : round(train_loss, 4),
                "train_acc"  : round(train_acc,  2),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in val_metrics.items()},
            })

            # ── MLflow: per-epoch train + val metrics ─────────
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "train/loss"   : round(train_loss, 4),
                        "train/acc"    : round(train_acc,  2),
                        "val/macro_f1" : round(val_metrics["macro_f1"], 4),
                        "val/auc_roc"  : round(val_metrics["auc_roc"],  4),
                        "val/fake_f1"  : round(val_metrics["fake_f1"],  4),
                        "val/real_f1"  : round(val_metrics["real_f1"],  4),
                    },
                    step=epoch + 1,
                )

            # Save best model if val Macro-F1 improved
            if val_metrics["macro_f1"] > self.best_f1:
                self.best_f1 = val_metrics["macro_f1"]
                self._save_best_model(val_metrics)
                logging.info(f"  🏆 New best Macro-F1 = {self.best_f1:.4f}")
            else:
                logging.info(f"  No improvement. Best = {self.best_f1:.4f}")

            # Save checkpoint (every epoch, keep latest 2)
            self._save_checkpoint(epoch)

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full Mamba training pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_mamba_training(self):
        """
        Execute the Bangla-Mamba training pipeline end-to-end (Steps 1–10).

        Opens a single MLflow run that spans both training and evaluation.
        After training completes, hands off to MambaEvaluate for
        Steps 7–10 (test evaluation, thesis experiment, results & summary).

        DagsHub credentials are read from env vars:
          MLFLOW_TRACKING_USERNAME — DagsHub username
          MLFLOW_TRACKING_PASSWORD — DagsHub access token / password
        """
        section("Bangla-Mamba Training — A100 40GB")

        self._set_seed()
        self._configure_device()

        # ── MLflow experiment setup ────────────────────────────
        mlflow.set_tracking_uri(settings.mamba_mlflow.tracking_uri)
        mlflow.set_experiment(settings.mamba_mlflow.experiment_name)
        logging.info(f"  MLflow tracking URI : {settings.mamba_mlflow.tracking_uri}")
        logging.info(f"  MLflow experiment   : {settings.mamba_mlflow.experiment_name}")

        with mlflow.start_run(run_name=settings.mamba_mlflow.run_name) as _run:
            logging.info(f"  MLflow run ID       : {_run.info.run_id}")

            # ── Log all hyper-parameters at run start ──────────
            mlflow.log_params({
                "tokenizer_name"    : self.tokenizer_name,
                "vocab_size"        : self.vocab_size,
                "d_model"           : self.d_model,
                "n_layer"           : self.n_layer,
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
                "optimizer_betas"   : "[0.9, 0.95]",
                "seed"              : self.seed,
            })
            mlflow.set_tags({
                "device"    : self.device,
                "framework" : "mamba-ssm",
                "task"      : "text-classification",
                "language"  : "bengali",
            })

            self.load_tokenizer()
            total_steps = self.load_datasets()
            start_epoch = self.build_model(total_steps)
            self.train(start_epoch)   # logs per-epoch metrics internally

            # ── Build training config snapshot for the evaluator ──
            training_config = {
                "model"          : "BanglaMamba-43M",
                "d_model"        : self.d_model,
                "n_layer"        : self.n_layer,
                "vocab_size"     : self.vocab_size,
                "max_length"     : self.max_length,
                "epochs"         : self.epochs,
                "batch_size"     : self.batch_size,
                "grad_accum"     : self.grad_accum,
                "effective_batch": self.batch_size * self.grad_accum,
                "lr"             : self.learning_rate,
                "precision"      : "BF16" if self.use_bf16 else "FP32",
                "class_weights"  : self.class_weights,
                "optimizer_betas": [0.9, 0.95],
            }

            # ── Delegate evaluation (Steps 7–10) to MambaEvaluate ──
            # MambaEvaluate runs inside this same active MLflow run;
            # it logs final test metrics, results JSON, and the best model.
            evaluator = MambaEvaluate()
            evaluator.initialize_mamba_evaluation(
                test_loader      = self.test_loader,
                tokenizer        = self.tokenizer,
                training_config  = training_config,
                training_history = self.history,
                best_val_f1      = self.best_f1,
            )
        # ── MLflow run closes automatically here ──────────────
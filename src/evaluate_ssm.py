"""
=============================================================
Bangla-Mamba Evaluation — Bangla Fake News Classification
=============================================================
Runs post-training evaluation steps on the best saved model:
  - STEP 7 · Final Test Evaluation  (best model on full test set)
  - STEP 8 · Thesis Experiment      (long vs short article F1)
  - STEP 9 · Save Results           (JSON + thesis table)
  - STEP 10 · Print Summary

PRE-REQUISITE:
  A completed training run with a best model saved at
  Artifacts/best_model/mamba/
=============================================================
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from src.utils.logger import log_filepath
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import Dataset
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
import mlflow
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import create_directory, section
from config.config import settings
from config.params import params

from src.ssm_model import BanglaMambaForClassification


class MambaEvaluate:
    """
    Post-training evaluation pipeline for Bangla-Mamba (SSM).

    Loads the best saved Mamba model from disk, runs final test-set evaluation,
    thesis experiment (long vs short article F1), persists JSON results,
    and prints a summary table.

    Designed to be called after MambaTrainer.initialize_mamba_training()
    has completed, or independently as a standalone evaluation run.
    """

    def __init__(self):
        # ── Paths (from config) ────────────────────────────────
        self.best_model_dir  = settings.mamba_evaluate.best_model_dir
        self.results_file    = settings.mamba_evaluate.results_file
        self.short_test_path = settings.mamba_evaluate.short_test_subset
        self.long_test_path  = settings.mamba_evaluate.long_test_subset

        # ── Hyper-parameters (from params) ─────────────────────
        self.tokenizer_name = params.mamba.tokenizer_name
        self.vocab_size     = params.mamba.vocab_size
        self.d_model        = params.mamba.d_model
        self.n_layer        = params.mamba.n_layer
        self.num_labels     = params.mamba.num_labels
        self.max_length     = params.mamba.max_length
        self.batch_size     = params.mamba.batch_size
        self.use_bf16       = params.mamba.use_bf16
        self.num_workers    = params.mamba.num_workers

        # Global setting
        self.seed = settings.seed

        # ── Runtime state ──────────────────────────────────────
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.model        = None
        self.tokenizer    = None
        self.eval_profiling: dict = {}   # populated by _evaluate(profile=True)

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
    # INTERNAL — Model inference helper
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        split_name: str = "test",
        profile: bool = False,
    ) -> dict:
        """
        Run inference on *loader* and return a metrics dict.

        When profile=True (used only for the full test set in evaluate_test):
          - Resets GPU peak VRAM stats before the loop.
          - Times every batch (with cuda.synchronize() for accuracy).
          - Stores profiling results in self.eval_profiling:
              eval_num_batches, eval_total_time_s,
              eval_avg_batch_ms, eval_p50_batch_ms, eval_p95_batch_ms,
              eval_max_batch_ms, eval_peak_vram_mb (CUDA only).
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        batch_times_ms: list[float] = []

        # ── Reset VRAM peak counter before the evaluation pass ──────────
        if profile and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"]

            # Synchronise before timing so previous async ops don’t bleed in
            if profile and self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.use_bf16):
                logits = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                )

            # Synchronise after forward so the timer captures GPU work
            if profile and self.device == "cuda":
                torch.cuda.synchronize()
            if profile:
                batch_times_ms.append((time.perf_counter() - t0) * 1000)

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

        # ── Collect profiling stats ─────────────────────────────────
        if profile and batch_times_ms:
            total_samples = len(all_labels)
            total_time_s = sum(batch_times_ms) / 1000
            throughput_samples = total_samples / total_time_s if total_time_s > 0 else 0.0
            throughput_tokens = (total_samples * self.max_length) / total_time_s if total_time_s > 0 else 0.0

            profiling: dict = {
                "eval_num_batches"              : len(batch_times_ms),
                "eval_total_time_s"             : round(total_time_s, 3),
                "eval_avg_batch_ms"             : round(float(np.mean(batch_times_ms)),              2),
                "eval_p50_batch_ms"             : round(float(np.percentile(batch_times_ms, 50)),    2),
                "eval_p95_batch_ms"             : round(float(np.percentile(batch_times_ms, 95)),    2),
                "eval_max_batch_ms"             : round(float(np.max(batch_times_ms)),               2),
                "eval_throughput_samples_per_s" : round(throughput_samples, 2),
                "eval_throughput_tokens_per_s"  : round(throughput_tokens, 2),
            }
            if self.device == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                profiling["eval_peak_vram_mb"] = round(peak_mb, 1)
            self.eval_profiling = profiling

            logging.info(
                f"  Eval profiling: total={profiling['eval_total_time_s']:.2f}s  "
                f"avg_batch={profiling['eval_avg_batch_ms']:.1f}ms  "
                f"throughput={profiling['eval_throughput_samples_per_s']:.1f} samples/s ({profiling['eval_throughput_tokens_per_s']:.1f} tokens/s)  "
                + (f"peak_VRAM={profiling.get('eval_peak_vram_mb', 0):.0f} MB"
                   if self.device == "cuda" else "")
            )

        return metrics

    # ──────────────────────────────────────────────────────────
    # STEP 7 — FINAL TEST EVALUATION
    # ──────────────────────────────────────────────────────────
    def evaluate_test(self, test_loader: DataLoader) -> dict:
        """
        Load the best saved model and evaluate it on the full test set.

        Args:
            test_loader: DataLoader for the test split.

        Returns:
            test_metrics dict with macro_f1, auc_roc, per-class P/R/F1.
        """
        section("STEP 7 · Final Test Evaluation — Best Model")
        logging.info(f"  Loading best model from {self.best_model_dir} ...")

        try:
            self.model = BanglaMambaForClassification.load(
                str(self.best_model_dir),
                device=self.device,
            )
        except Exception as e:
            raise CustomException(e, sys)

        test_metrics = self._evaluate(test_loader, split_name="test", profile=True)
        return test_metrics

    # ──────────────────────────────────────────────────────────
    # STEP 8 — THESIS EXPERIMENT: Long vs Short Article F1
    # ──────────────────────────────────────────────────────────
    def thesis_experiment(self, tokenizer) -> dict:
        """
        Evaluate the best model on long (>512 tokens) and short (≤512 tokens)
        test subsets to compare with BanglaBERT's truncation cost.

        Args:
            tokenizer: The tokenizer (passed from MambaTrainer).

        Returns:
            subset_results dict keyed by subset label.
        """
        section("STEP 8 · Thesis Experiment — Long vs Short Article F1")
        logging.info(f"  Mamba-{self.max_length} is trained with max_length={self.max_length}.")
        logging.info("  This shows how much F1 changes on long vs short articles.\n")

        self.tokenizer = tokenizer
        subset_results = {}
        subsets = [
            ("short_test  (≤512 tokens)", self.short_test_path),
            ("long_test   (>512 tokens)", self.long_test_path),
        ]

        for label, csv_path in subsets:
            if not Path(csv_path).exists():
                logging.warning(f"  ⚠️  {csv_path} not found — skipping")
                continue

            try:
                sub_df = pd.read_csv(csv_path)
            except Exception as e:
                raise CustomException(e, sys)

            logging.info(f"  Evaluating on {label}: {len(sub_df):,} articles")

            sub_ds = Dataset.from_pandas(
                sub_df[["content", "label"]], preserve_index=False
            )

            def tok_sub(examples):
                return self.tokenizer(
                    examples["content"],
                    max_length     = self.max_length,
                    truncation     = True,
                    padding        = "max_length",
                    return_tensors = None,
                )

            remove_cols = [c for c in sub_ds.column_names if c != "label"]
            sub_ds = sub_ds.map(
                tok_sub, batched=True, batch_size=512,
                remove_columns=remove_cols,
            )
            sub_ds.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"]
            )

            sub_loader = DataLoader(
                sub_ds,
                batch_size  = self.batch_size * 2,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = True,
            )

            sub_metrics = self._evaluate(sub_loader, split_name=label)
            subset_results[label] = sub_metrics

        return subset_results

    # ──────────────────────────────────────────────────────────
    # STEP 9 — PRINT THESIS TABLE + SAVE JSON RESULTS
    # ──────────────────────────────────────────────────────────
    def save_results(
        self,
        test_metrics: dict,
        subset_results: dict,
        training_config: dict,
        training_history: list,
        best_val_f1: float,
        best_epoch: int = None,
    ):
        """
        Print thesis comparison table and persist all results to JSON.
        """
        section("STEP 9 · Results Summary & Save")

        # ── Thesis table ──────────────────────────────────────
        logging.info("\n  ┌──────────────────────────────────────────────────┐")
        logging.info("  │  THESIS TABLE: Bangla-Mamba Performance by Length│")
        logging.info("  ├──────────────────────────────────────────────────┤")
        logging.info(f"  │  {'Subset':<28} {'Macro-F1':>10} {'AUC-ROC':>10} │")
        logging.info("  ├──────────────────────────────────────────────────┤")
        logging.info(f"  │  {'Full test set':<28} "
                      f"{test_metrics['macro_f1']:>10.4f} "
                      f"{test_metrics['auc_roc']:>10.4f} │")
        for lbl, m in subset_results.items():
            logging.info(f"  │  {lbl:<28} "
                          f"{m['macro_f1']:>10.4f} "
                          f"{m['auc_roc']:>10.4f} │")
        logging.info("  └──────────────────────────────────────────────────┘")

        # ── Save JSON ─────────────────────────────────────────
        create_directory(self.results_file.parent)

        # Calculate overall peak training VRAM if available in history
        train_peaks = [
            h["train_peak_vram_mb"]
            for h in training_history
            if isinstance(h, dict) and "train_peak_vram_mb" in h
        ]
        overall_train_peak = max(train_peaks) if train_peaks else None

        results = {
            "config"                       : training_config,
            "training_history"             : training_history,
            "best_val_macro_f1"            : round(best_val_f1, 4),
            "best_epoch"                   : best_epoch,
            "overall_training_peak_vram_mb": overall_train_peak,
            "test_metrics"                 : test_metrics,
            "subset_results"               : subset_results,
            "profiling"                    : self.eval_profiling,   # VRAM + batch timing
        }
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"  Results saved → {self.results_file}")

    # ──────────────────────────────────────────────────────────
    # STEP 10 — FINAL SUMMARY
    # ──────────────────────────────────────────────────────────
    def print_summary(
        self,
        test_metrics: dict,
        training_history: list,
        best_val_f1: float,
        best_epoch: int = None,
    ):
        """
        Print a final boxed summary of Bangla-Mamba results.
        """
        section("Evaluation Complete ✅")
        total_train_min = sum(h.get("time_min", 0) for h in training_history)
        epoch_str = str(best_epoch) if best_epoch is not None else "N/A"
        throughput_samp = self.eval_profiling.get("eval_throughput_samples_per_s", 0.0)
        throughput_tok = self.eval_profiling.get("eval_throughput_tokens_per_s", 0.0)
        logging.info(f"""
  ┌──────────────────────────────────────────────────┐
  │            BANGLA-MAMBA RESULTS SUMMARY          │
  ├──────────────────────────────────────────────────┤
  │  Best Val  Macro-F1  : {best_val_f1:<10.4f}                │
  │  Best Epoch          : {epoch_str:<10}                │
  │  Test      Macro-F1  : {test_metrics['macro_f1']:<10.4f}                │
  │  Test      AUC-ROC   : {test_metrics['auc_roc']:<10.4f}                │
  │  Test      MCC       : {test_metrics['mcc']:<10.4f}                │
  │  Test      PR-AUC    : {test_metrics['pr_auc']:<10.4f}                │
  │  Test      ECE       : {test_metrics['ece']:<10.4f}                │
  │  Test      W-F1      : {test_metrics['weighted_f1']:<10.4f}                │
  │  Test      Fake-F1   : {test_metrics['fake_f1']:<10.4f}                │
  │  Test      Real-F1   : {test_metrics['real_f1']:<10.4f}                │
  ├──────────────────────────────────────────────────┤
  │  Total training time : {total_train_min:<6.1f} min                │
  │  Precision used      : {'BF16 (A100 native)' if self.use_bf16 else 'FP32':<25} │
  │  Throughput (samp/s) : {throughput_samp:<10.2f}                │
  │  Throughput (tok/s)  : {throughput_tok:<10.2f}                │
  ├──────────────────────────────────────────────────┤
  │  Best model → {str(self.best_model_dir):<33} │
  │  Results   → {str(self.results_file):<33} │
  └──────────────────────────────────────────────────┘
""")

    # ──────────────────────────────────────────────────────────
    # MLflow logging
    # ──────────────────────────────────────────────────────────
    def _log_to_mlflow(
        self,
        test_metrics: dict,
        subset_results: dict,
        best_val_f1: float,
        best_epoch: int = None,
    ):
        """
        Log final evaluation artefacts to the currently active MLflow run.
        """
        if not mlflow.active_run():
            logging.warning("  ⚠️  No active MLflow run — skipping MLflow logging.")
            return

        section("MLflow · Logging Evaluation Results")

        # ── 1. Final scalar metrics ──────────────────────────────────
        metrics_to_log = {
            "best_val/macro_f1"      : round(best_val_f1, 4),
            "test/macro_f1"          : round(test_metrics["macro_f1"],          4),
            "test/auc_roc"           : round(test_metrics["auc_roc"],           4),
            "test/mcc"               : round(test_metrics["mcc"],               4),
            "test/pr_auc"            : round(test_metrics["pr_auc"],            4),
            "test/ece"               : round(test_metrics["ece"],               4),
            "test/weighted_precision": round(test_metrics["weighted_precision"], 4),
            "test/weighted_recall"   : round(test_metrics["weighted_recall"],   4),
            "test/weighted_f1"       : round(test_metrics["weighted_f1"],       4),
            "test/fake_precision"    : round(test_metrics["fake_precision"],    4),
            "test/fake_recall"       : round(test_metrics["fake_recall"],       4),
            "test/fake_f1"           : round(test_metrics["fake_f1"],           4),
            "test/real_precision"    : round(test_metrics["real_precision"],    4),
            "test/real_recall"       : round(test_metrics["real_recall"],       4),
            "test/real_f1"           : round(test_metrics["real_f1"],           4),
        }
        if best_epoch is not None:
            metrics_to_log["best_val/epoch"] = best_epoch
        mlflow.log_metrics(metrics_to_log)
        logging.info("  ✓ Test metrics logged to MLflow")

        # ── 2. Thesis experiment subset metrics ──────────────────────
        for label, m in subset_results.items():
            prefix = "short_test" if "\u2264512" in label else "long_test"
            mlflow.log_metrics({
                f"{prefix}/macro_f1" : round(m["macro_f1"], 4),
                f"{prefix}/auc_roc"  : round(m["auc_roc"],  4),
                f"{prefix}/fake_f1"  : round(m["fake_f1"],  4),
                f"{prefix}/real_f1"  : round(m["real_f1"],  4),
            })
        logging.info("  ✓ Subset metrics logged to MLflow")

        # ── 2b. Profiling metrics (VRAM + batch timing) ───────────────
        if self.eval_profiling:
            mlflow.log_metrics({
                f"eval_profile/{k}": v
                for k, v in self.eval_profiling.items()
                if isinstance(v, (int, float))
            })
            logging.info("  ✓ Profiling metrics logged to MLflow")

        # ── 3. Results JSON artifact ────────────────────────────────
        if self.results_file.exists():
            mlflow.log_artifact(str(self.results_file), artifact_path="results")
            logging.info(f"  ✓ Results JSON logged  → results/{self.results_file.name}")

        # ── 3b. Run log file artifact ───────────────────────────────
        if os.path.exists(log_filepath):
            mlflow.log_artifact(log_filepath, artifact_path="logs")
            logging.info(f"  ✓ Run log file logged  → logs/{os.path.basename(log_filepath)}")

        # ── 4. Best model ──
        if settings.mamba_mlflow.log_model and self.best_model_dir.exists():
            logging.info("  📦 Logging best model to MLflow (Mamba files) ...")
            try:
                mlflow.log_artifacts(
                    str(self.best_model_dir),
                    artifact_path="best_model",
                )
                logging.info("  ✓ Best model logged to MLflow under 'best_model/'")
            except Exception as e:
                logging.warning(f"  ⚠️  mlflow.log_artifacts (best_model) failed: {e}")
        elif not settings.mamba_mlflow.log_model:
            logging.info("  ⏩ Model upload skipped (mlflow.log_model=False)")

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full evaluation pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_mamba_evaluation(
        self,
        test_loader: DataLoader,
        tokenizer,
        training_config: dict,
        training_history: list,
        best_val_f1: float,
    ):
        """
        Execute the full Bangla-Mamba evaluation pipeline end-to-end.

        Args:
            test_loader:      DataLoader for the test split (from MambaTrainer).
            tokenizer:        Tokenizer (from MambaTrainer).
            training_config:  Config snapshot dict (from MambaTrainer).
            training_history: Per-epoch history list (from MambaTrainer).
            best_val_f1:      Best validation Macro-F1 (from MambaTrainer).
        """
        # Determine the best epoch from training history
        best_epoch = None
        if training_history:
            best_idx = np.argmax([h.get("macro_f1", 0.0) for h in training_history])
            best_epoch = int(training_history[best_idx].get("epoch"))

        test_metrics   = self.evaluate_test(test_loader)
        subset_results = self.thesis_experiment(tokenizer)
        self.save_results(
            test_metrics,
            subset_results,
            training_config,
            training_history,
            best_val_f1,
            best_epoch,
        )
        self.print_summary(test_metrics, training_history, best_val_f1, best_epoch)
        self._log_to_mlflow(test_metrics, subset_results, best_val_f1, best_epoch)

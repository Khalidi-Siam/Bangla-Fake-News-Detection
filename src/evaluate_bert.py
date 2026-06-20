"""
=============================================================
BanglaBERT Evaluation — Bangla Fake News Classification
=============================================================
Runs post-training evaluation steps on the best saved model:
  - STEP 7 · Final Test Evaluation  (best model on full test set)
  - STEP 8 · Thesis Experiment      (long vs short article F1)
  - STEP 9 · Save Results           (JSON + thesis table)
  - STEP 10 · Print Summary

PRE-REQUISITE:
  A completed fine-tuning run with a best model saved at
  Artifacts/best_model/banglabert/
=============================================================
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
import mlflow
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import create_directory, section
from config.config import settings
from config.params import params


class BertEvaluate:
    """
    Post-training evaluation pipeline for BanglaBERT.

    Loads the best saved model from disk, runs final test-set evaluation,
    thesis experiment (long vs short article F1), persists JSON results,
    and prints a summary table.

    Designed to be called after BertFineTune.initialize_bert_finetuning()
    has completed, or independently as a standalone evaluation run.
    """

    def __init__(self):
        # ── Paths (from config) ────────────────────────────────
        self.best_model_dir  = settings.bert_evaluate.best_model_dir
        self.results_file    = settings.bert_evaluate.results_file
        self.short_test_path = settings.bert_evaluate.short_test_subset
        self.long_test_path  = settings.bert_evaluate.long_test_subset

        # ── Hyper-parameters (from params) ─────────────────────
        self.model_name  = params.bert.model_name
        self.num_labels  = params.bert.num_labels
        self.max_length  = params.bert.max_length
        self.batch_size  = params.bert.batch_size
        self.use_bf16    = params.bert.use_bf16
        self.num_workers = params.bert.num_workers

        # Global setting
        self.seed = settings.seed

        # ── Runtime state ──────────────────────────────────────
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.model        = None
        self.tokenizer    = None
        self.eval_profiling: dict = {}   # populated by _evaluate(profile=True)

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
                outputs = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                )

            # Synchronise after forward so the timer captures GPU work
            if profile and self.device == "cuda":
                torch.cuda.synchronize()
            if profile:
                batch_times_ms.append((time.perf_counter() - t0) * 1000)

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

        # ── Collect profiling stats ─────────────────────────────────
        if profile and batch_times_ms:
            profiling: dict = {
                "eval_num_batches"   : len(batch_times_ms),
                "eval_total_time_s"  : round(sum(batch_times_ms) / 1000, 3),
                "eval_avg_batch_ms"  : round(float(np.mean(batch_times_ms)),              2),
                "eval_p50_batch_ms"  : round(float(np.percentile(batch_times_ms, 50)),    2),
                "eval_p95_batch_ms"  : round(float(np.percentile(batch_times_ms, 95)),    2),
                "eval_max_batch_ms"  : round(float(np.max(batch_times_ms)),               2),
            }
            if self.device == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                profiling["eval_peak_vram_mb"] = round(peak_mb, 1)
            self.eval_profiling = profiling

            logging.info(
                f"  Eval profiling: total={profiling['eval_total_time_s']:.2f}s  "
                f"avg_batch={profiling['eval_avg_batch_ms']:.1f}ms  "
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
            test_loader: DataLoader for the test split (built by BertFineTune).

        Returns:
            test_metrics dict with macro_f1, auc_roc, per-class P/R/F1.
        """
        from src.utils.common import section
        section("STEP 7 · Final Test Evaluation — Best Model")
        logging.info(f"  Loading best model from {self.best_model_dir} ...")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.best_model_dir),
                num_labels=self.num_labels,
            ).to(self.device)
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
        test subsets to quantify the truncation cost of BanglaBERT.

        Args:
            tokenizer: The BanglaBERT tokenizer (passed from BertFineTune).

        Returns:
            subset_results dict keyed by subset label.
        """
        from src.utils.common import section
        section("STEP 8 · Thesis Experiment — Long vs Short Article F1")
        logging.info("  BanglaBERT truncates long articles (>512 tokens).")
        logging.info("  This shows how much F1 degrades on those articles.")
        logging.info("  Mamba-768 will be compared against these numbers.\n")

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
    ):
        """
        Print thesis comparison table and persist all results to JSON.

        Args:
            test_metrics:     Metrics dict from evaluate_test().
            subset_results:   Subset metrics dict from thesis_experiment().
            training_config:  Config snapshot from BertFineTune (model, LR, etc.).
            training_history: Per-epoch history list from BertFineTune.
            best_val_f1:      Best validation Macro-F1 from BertFineTune.
        """
        from src.utils.common import section
        section("STEP 9 · Results Summary & Save")

        # ── Thesis table ──────────────────────────────────────
        logging.info("\n  ┌──────────────────────────────────────────────────┐")
        logging.info("  │  THESIS TABLE: BanglaBERT Performance by Length │")
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
        logging.info("  NOTE: long_test F1 drop = BanglaBERT truncation cost")
        logging.info("        Mamba-768 should close this gap.")

        # ── Save JSON ─────────────────────────────────────────
        create_directory(self.results_file.parent)
        results = {
            "config"            : training_config,
            "training_history"  : training_history,
            "best_val_macro_f1" : round(best_val_f1, 4),
            "test_metrics"      : test_metrics,
            "subset_results"    : subset_results,
            "profiling"         : self.eval_profiling,   # VRAM + batch timing
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
    ):
        """
        Print a final boxed summary of BanglaBERT results.

        Args:
            test_metrics:     Metrics dict from evaluate_test().
            training_history: Per-epoch history list from BertFineTune.
            best_val_f1:      Best validation Macro-F1 from BertFineTune.
        """
        from src.utils.common import section
        section("Evaluation Complete ✅")
        total_train_min = sum(h.get("time_min", 0) for h in training_history)
        logging.info(f"""
  ┌──────────────────────────────────────────────────┐
  │            BANGLABERT RESULTS SUMMARY            │
  ├──────────────────────────────────────────────────┤
  │  Best Val  Macro-F1  : {best_val_f1:<10.4f}                │
  │  Test      Macro-F1  : {test_metrics['macro_f1']:<10.4f}                │
  │  Test      AUC-ROC   : {test_metrics['auc_roc']:<10.4f}                │
  │  Test      Fake-F1   : {test_metrics['fake_f1']:<10.4f}                │
  │  Test      Real-F1   : {test_metrics['real_f1']:<10.4f}                │
  ├──────────────────────────────────────────────────┤
  │  Total training time : {total_train_min:<6.1f} min                  │
  │  Precision used      : {'BF16 (A100 native)' if self.use_bf16 else 'FP32':<25}   │
  ├──────────────────────────────────────────────────┤
  │  Best model → {str(self.best_model_dir):<33} │
  │  Results   → {str(self.results_file):<33} │
  └──────────────────────────────────────────────────┘

  NEXT: Use these numbers as the baseline target
        when training Mamba-768.
""")

    # ──────────────────────────────────────────────────────────
    # MLflow logging (called inside the active run from finetune_bert)
    # ──────────────────────────────────────────────────────────
    def _log_to_mlflow(
        self,
        test_metrics: dict,
        subset_results: dict,
        best_val_f1: float,
    ):
        """
        Log final evaluation artefacts to the currently active MLflow run.

        Operates inside the run opened by BertFineTune.initialize_bert_finetuning().
        Three things are logged:
          1. Scalar metrics  — test set + thesis experiment subsets.
          2. Results JSON    — logged as an MLflow artifact file.
          3. Best model      — reloaded from disk (self.model) then logged via
                               mlflow.transformers.log_model, guaranteeing
                               exactly the saved weights are stored in MLflow.
        """
        if not mlflow.active_run():
            logging.warning("  ⚠️  No active MLflow run — skipping MLflow logging.")
            return

        section("MLflow · Logging Evaluation Results")

        # ── 1. Final scalar metrics ──────────────────────────────────
        mlflow.log_metrics({
            "best_val/macro_f1"   : round(best_val_f1, 4),
            "test/macro_f1"       : round(test_metrics["macro_f1"],       4),
            "test/auc_roc"        : round(test_metrics["auc_roc"],        4),
            "test/fake_precision" : round(test_metrics["fake_precision"],  4),
            "test/fake_recall"    : round(test_metrics["fake_recall"],     4),
            "test/fake_f1"        : round(test_metrics["fake_f1"],         4),
            "test/real_precision" : round(test_metrics["real_precision"],  4),
            "test/real_recall"    : round(test_metrics["real_recall"],     4),
            "test/real_f1"        : round(test_metrics["real_f1"],         4),
        })
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
                if isinstance(v, (int, float))   # guard against any str values
            })
            logging.info("  ✓ Profiling metrics logged to MLflow")

        # ── 3. Results JSON artifact ────────────────────────────────
        if self.results_file.exists():
            mlflow.log_artifact(str(self.results_file), artifact_path="results")
            logging.info(f"  ✓ Results JSON logged  → results/{self.results_file.name}")

        # ── 4. Best model (upload saved HF directory as MLflow artifacts) ──
        #
        # The best model is already saved to disk in HuggingFace format by
        # BertFineTune._save_best_model().  We use mlflow.log_artifacts() to
        # upload the directory directly — no re-serialisation, no dependency
        # scan, and no torchvision false-positive import errors.
        # Files (config.json, model.safetensors, tokenizer files, etc.) can
        # be reloaded later with:
        #   AutoModelForSequenceClassification.from_pretrained("<run>/best_model")
        if settings.mlflow.log_model and self.best_model_dir.exists():
            logging.info("  📦 Logging best model to MLflow (HuggingFace files) ...")
            try:
                mlflow.log_artifacts(
                    str(self.best_model_dir),
                    artifact_path="best_model",
                )
                logging.info("  ✓ Best model logged to MLflow under 'best_model/'")
            except Exception as e:
                # Non-fatal: log a warning but don't crash the pipeline
                logging.warning(f"  ⚠️  mlflow.log_artifacts (best_model) failed: {e}")
        elif not settings.mlflow.log_model:
            logging.info("  ⏩ Model upload skipped (mlflow.log_model=False)")

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full evaluation pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_bert_evaluation(
        self,
        test_loader: DataLoader,
        tokenizer,
        training_config: dict,
        training_history: list,
        best_val_f1: float,
    ):
        """
        Execute the full BanglaBERT evaluation pipeline end-to-end.

        Args:
            test_loader:      DataLoader for the test split (from BertFineTune).
            tokenizer:        BanglaBERT tokenizer (from BertFineTune).
            training_config:  Config snapshot dict (from BertFineTune).
            training_history: Per-epoch history list (from BertFineTune).
            best_val_f1:      Best validation Macro-F1 (from BertFineTune).
        """
        test_metrics   = self.evaluate_test(test_loader)
        subset_results = self.thesis_experiment(tokenizer)
        self.save_results(
            test_metrics,
            subset_results,
            training_config,
            training_history,
            best_val_f1,
        )
        self.print_summary(test_metrics, training_history, best_val_f1)

        # ── MLflow: log metrics + artifacts + best model ──────────────
        # Runs inside the active MLflow run opened by BertFineTune.
        # self.model is already the best model reloaded from disk;
        # self.tokenizer is set by thesis_experiment() above.
        self._log_to_mlflow(test_metrics, subset_results, best_val_f1)

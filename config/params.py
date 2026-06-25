from typing import List
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────
# BanglaBERT Fine-Tuning Hyper-Parameters
# ─────────────────────────────────────────────────────────────
class BertFineTuneParams(BaseModel):
    """All tunable hyper-parameters for BanglaBERT fine-tuning."""

    # ── Model ──────────────────────────────────────────────────
    model_name: str         = "csebuetnlp/banglabert"
    num_labels: int         = 2
    max_length: int         = 512          # BanglaBERT hard architectural limit

    # ── Training ───────────────────────────────────────────────
    epochs: int             = 5
    batch_size: int         = 32           # A100 40GB comfortably handles this at 512 tokens
    grad_accum: int         = 2            # effective batch = batch_size × grad_accum
    learning_rate: float    = 2e-5
    weight_decay: float     = 0.01
    warmup_pct: float       = 0.1          # fraction of total steps used for LR warm-up
    max_grad_norm: float    = 1.0

    # ── Precision ──────────────────────────────────────────────
    # BF16 is faster & more stable than FP16 on A100; no GradScaler needed
    use_bf16: bool          = True

    # ── Class weights (derived from preprocessing log) ─────────
    class_weights: List[float] = [3.0295, 0.5988]   # [Fake(0), Real(1)]

    # ── DataLoader ─────────────────────────────────────────────
    num_workers: int        = 4

    @field_validator("warmup_pct")
    @classmethod
    def warmup_must_be_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("warmup_pct must be strictly between 0 and 1")
        return v

    @field_validator("class_weights")
    @classmethod
    def must_have_two_weights(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("class_weights must contain exactly 2 values [fake, real]")
        return v


# ─────────────────────────────────────────────────────────────
# Bangla-Mamba (SSM) Training Hyper-Parameters
# ─────────────────────────────────────────────────────────────
class MambaTrainParams(BaseModel):
    """All tunable hyper-parameters for Bangla-Mamba (SSM) training."""

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer_name: str     = "csebuetnlp/banglabert"

    # ── Model ──────────────────────────────────────────────────
    vocab_size: int         = 32000   # banglabert vocab — confirmed in preprocessing
    d_model: int            = 512
    n_layer: int            = 16
    num_labels: int         = 2
    max_length: int         = 1024     # Mamba context window (vs BERT 512)

    # ── Training ───────────────────────────────────────────────
    epochs: int             = 5
    batch_size: int         = 32       # Mamba at 768 tokens is more memory-hungry
                                      # than BERT at 512. 8 is safe on A100-40GB.
    grad_accum: int         = 2       # effective batch = 8 × 8 = 64 (same as BERT)
    learning_rate: float    = 1e-3    # Mamba trains from scratch → higher LR than BERT
    weight_decay: float     = 0.01
    warmup_pct: float       = 0.1     # 10% warmup then cosine decay
    max_grad_norm: float    = 1.0

    # ── Precision ──────────────────────────────────────────────
    # BF16 on A100 — no GradScaler needed
    use_bf16: bool          = True

    # ── Class weights (derived from preprocessing log) ─────────
    class_weights: List[float] = [3.0295, 0.5988]   # [Fake(0), Real(1)]

    # ── DataLoader ─────────────────────────────────────────────
    num_workers: int        = 4

    @field_validator("warmup_pct")
    @classmethod
    def warmup_must_be_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("warmup_pct must be strictly between 0 and 1")
        return v

    @field_validator("class_weights")
    @classmethod
    def must_have_two_weights_mamba(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("class_weights must contain exactly 2 values [fake, real]")
        return v


# ─────────────────────────────────────────────────────────────
# Params Settings  (env-override support, mirrors Settings)
# ─────────────────────────────────────────────────────────────
class ParamSettings(BaseSettings):
    bert: BertFineTuneParams = BertFineTuneParams()
    mamba: MambaTrainParams  = MambaTrainParams()

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_file_required=False,
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


params = ParamSettings()

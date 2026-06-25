"""
=============================================================
predict.py — Bangla Fake News Prediction Script
=============================================================
Supports  : BanglaBERT  (fine-tuned)  or  Bangla-Mamba (SSM)
Input     : title (headline) + news body (content)
            → merged as  "title [SEP] body"  (same as training)
Cleaning  : same clean_bangla() used in data_ingestion.py
Output    : Fake / Real label + class probabilities

Mamba inference:
  • CPU only        → HuggingFace MambaModel  (pure PyTorch)
                      loads from  Artifacts/best_model/mamba_1024_hf/
                      (generate with modal_utils/convert_mamba_to_hf.py)
=============================================================
Usage (two-variable mode — UI variables):
    Set  `input_title` and `input_body` at the bottom of this
    file, then run:
        python predict.py
=============================================================
"""

import re
import sys
import json
import unicodedata
from pathlib import Path
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Project root on sys.path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from config.config import settings
from config.params import params


# =============================================================
# CLEANING (identical to data_ingestion.py → clean_bangla)
# =============================================================
def clean_bangla(text: str) -> str:
    """
    Cleans Bangla text the same way as done during preprocessing:
      - Strip HTML tags
      - Remove URLs
      - Normalize unicode to NFC  (fixes overlapping Bangla chars)
      - Collapse whitespace
      - Preserve all punctuation  (!, ?, ... are fake-news signals)
      - Preserves [SEP] separator token
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)           # strip HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = unicodedata.normalize('NFC', text)        # NFC unicode normalization
    text = re.sub(r'\s+', ' ', text).strip()         # collapse whitespace
    return text


# =============================================================
# INPUT PREPROCESSING
# =============================================================
def build_input_text(title: str, body: str) -> str:
    """
    Merges headline + body with [SEP] separator — exactly as done
    in DataIngestion.merge_headline():
        content = headline + ' [SEP] ' + content
    Then applies the same clean_bangla() cleaning pipeline.
    """
    title = title.strip() if title else ""
    body  = body.strip()  if body  else ""

    if title:
        merged = f"{title} [SEP] {body}"
    else:
        merged = body  # fall back to body only (mirrors ingestion logic)

    return clean_bangla(merged)


# =============================================================
# SHARED LABEL MAP
# =============================================================
LABEL_MAP = {0: "Fake 🔴", 1: "Real ✅"}


# =============================================================
# HuggingFace Mamba CPU Model
# (used when CUDA is not available)
# =============================================================
class _ClassificationHead(nn.Module):
    """
    Exact copy of ClassificationHead from ssm_model.py.
    Kept here so predict.py has zero dependency on mamba-ssm.
    """
    def __init__(self, d_model: int, head_hidden_dim: int,
                 head_dropout: float, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BanglaMambaHFForClassification(nn.Module):
    """
    CPU-compatible Bangla-Mamba classifier.
    Backbone  : transformers.MambaModel  (pure PyTorch — no CUDA kernels)
    Head      : same 2-layer MLP used during training
    Pooling   : mean over non-pad positions  (same as training)

    Weights are loaded from the HF-converted folder produced by
    modal_utils/convert_mamba_to_hf.py.
    """

    def __init__(self, hf_config, mamba_cfg: dict):
        super().__init__()
        from transformers import MambaModel
        self.backbone = MambaModel(hf_config)
        self.head     = _ClassificationHead(
            d_model        = mamba_cfg["d_model"],
            head_hidden_dim= mamba_cfg["head_hidden_dim"],
            head_dropout   = mamba_cfg["head_dropout"],
            num_labels     = mamba_cfg["num_labels"],
        )

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor | None = None,
    ) -> torch.Tensor:
        # HF MambaModel returns a BaseModelOutputWithNoAttention;
        # last_hidden_state has shape (B, L, D)
        hidden = self.backbone(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state   # (B, L, D)

        # Mean pool over non-pad positions (same as training)
        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)                      # (B, D)
            count  = mask.sum(dim=1).clamp(min=1e-9)                 # (B, 1)
            pooled = summed / count                                   # (B, D)
        else:
            pooled = hidden.mean(dim=1)                               # (B, D)

        return self.head(pooled)                                      # (B, 2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BanglaMambaHFForClassification":
        """
        Load from a HF-converted folder (output of convert_mamba_to_hf.py).
        Expects:
            <path>/config.json          — HF MambaConfig
            <path>/mamba_config.json    — head hyper-params
            <path>/model_weights.pt     — remapped state dict
        """
        from transformers import MambaConfig
        hf_config  = MambaConfig.from_pretrained(path)
        with open(f"{path}/mamba_config.json") as f:
            mamba_cfg = json.load(f)

        model = cls(hf_config, mamba_cfg)

        state = torch.load(f"{path}/model_weights.pt", map_location=device)
        # Strip leading "backbone." / "head." prefixes that are already
        # expected by our module — just load directly
        model.load_state_dict(state, strict=True)
        return model.to(device)


# =============================================================
# BERT PREDICTOR
# =============================================================
class BertPredictor:
    """
    Loads the best BanglaBERT model from
    Artifacts/best_model/banglabert/ and predicts on a single text.
    Works on CPU.
    """

    def __init__(self, device: str = "cpu"):
        self.best_model_dir = settings.bert_finetune.best_model_dir
        self.max_length     = params.bert.max_length
        self.device         = "cpu"
        self.tokenizer      = None
        self.model          = None

    def load(self):
        """Load tokenizer + model weights from best_model_dir."""
        if not self.best_model_dir.exists():
            raise FileNotFoundError(
                f"BanglaBERT best model not found at '{self.best_model_dir}'.\n"
                "Run the fine-tuning pipeline first (finetune_bert.py)."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.best_model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(self.best_model_dir)
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs   = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs     = torch.softmax(outputs.logits.float(), dim=-1)[0]
        pred_id   = int(probs.argmax().item())
        prob_fake = float(probs[0].item())
        prob_real = float(probs[1].item())

        return {
            "label"     : LABEL_MAP[pred_id],
            "label_id"  : pred_id,
            "prob_fake" : prob_fake,
            "prob_real" : prob_real,
            "confidence": max(prob_fake, prob_real),
        }


# =============================================================
# MAMBA PREDICTOR  — CPU (HF MambaModel)
# =============================================================
class MambaPredictor:
    """
    Mamba predictor for CPU (HF MambaModel).
    Loads from: Artifacts/best_model/mamba_1024_hf/
    """

    # Paths
    CPU_MODEL_DIR = settings.mamba_train.best_model_dir.parent / "mamba_1024_hf"  # HF weights

    def __init__(self, device: str = "cpu"):
        self.max_length = params.mamba.max_length
        self.device     = "cpu"
        self.tokenizer  = None
        self.model      = None
        self._backend   = "hf_cpu"

    def load(self):
        self._load_hf_cpu()

    # ── CPU backend ───────────────────────────────────────────
    def _load_hf_cpu(self):
        """Load HF-converted weights (pure PyTorch, CPU-safe)."""
        model_dir = self.CPU_MODEL_DIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"[Mamba-HF] Converted model not found at '{model_dir}'.\n"
                "Run the conversion script first:\n"
                "    modal run modal_utils/convert_mamba_to_hf.py\n"
                "Then download the folder to Artifacts/best_model/mamba_1024_hf/"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = BanglaMambaHFForClassification.load(
            str(model_dir), device=self.device
        )
        self.model.eval()

    # ── Inference ─────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits    = self.model(input_ids, attention_mask)
        probs     = torch.softmax(logits.float(), dim=-1)[0]
        pred_id   = int(probs.argmax().item())
        prob_fake = float(probs[0].item())
        prob_real = float(probs[1].item())

        return {
            "label"     : LABEL_MAP[pred_id],
            "label_id"  : pred_id,
            "prob_fake" : prob_fake,
            "prob_real" : prob_real,
            "confidence": max(prob_fake, prob_real),
        }

    @property
    def backend_name(self) -> str:
        return "Bangla-Mamba (HuggingFace / CPU)"


# =============================================================
# MAIN PREDICT FUNCTION
# =============================================================
def predict_news(
    title        : str,
    body         : str,
    model_choice : str       = "bert",   # "bert" or "mamba"
    device       : str       = "cpu",
) -> dict:
    """
    End-to-end prediction: preprocess → load model → predict.

    Args:
        title        : News headline / title.
        body         : News body / content.
        model_choice : "bert"  → BanglaBERT  (CPU)
                       "mamba" → Bangla-Mamba (CPU)
        device       : Ignored (always runs on CPU).

    Returns:
        dict with keys:
            label, label_id, prob_fake, prob_real, confidence,
            cleaned_input, model_used, token_count, backend
    """
    model_choice = model_choice.strip().lower()
    if model_choice not in ("bert", "mamba"):
        raise ValueError(f"model_choice must be 'bert' or 'mamba', got: '{model_choice}'")

    # ── 1. Preprocess (merge + clean) ─────────────────────────
    cleaned_text = build_input_text(title, body)
    if not cleaned_text:
        raise ValueError("Cleaned input is empty. Provide a non-empty title or body.")

    # ── 2. Load predictor ─────────────────────────────────────
    if model_choice == "bert":
        predictor = BertPredictor(device="cpu")
        backend   = "BanglaBERT (HuggingFace)"
    else:
        predictor = MambaPredictor(device="cpu")
        predictor.load()              # load first to resolve backend name
        backend = predictor.backend_name
        # token count + prediction below
        token_count = len(predictor.tokenizer.encode(cleaned_text, truncation=False))
        result = predictor.predict(cleaned_text)
        result.update({
            "cleaned_input": cleaned_text,
            "model_used"   : "Bangla-Mamba",
            "token_count"  : token_count,
            "backend"      : backend,
        })
        return result

    # BERT path (no early return above)
    predictor.load()
    token_count = len(predictor.tokenizer.encode(cleaned_text, truncation=False))
    result      = predictor.predict(cleaned_text)
    result.update({
        "cleaned_input": cleaned_text,
        "model_used"   : "BanglaBERT",
        "token_count"  : token_count,
        "backend"      : backend,
    })
    return result


# =============================================================
# PRETTY PRINT
# =============================================================
def print_result(result: dict):
    """Pretty-print the prediction result."""
    print("\n" + "=" * 60)
    print(f"  MODEL       : {result['model_used']}")
    print(f"  BACKEND     : {result['backend']}")
    print(f"  PREDICTION  : {result['label']}")
    print(f"  Confidence  : {result['confidence'] * 100:.2f}%")
    print(f"  Prob Fake   : {result['prob_fake'] * 100:.2f}%")
    print(f"  Prob Real   : {result['prob_real'] * 100:.2f}%")
    print(f"  Token count : {result['token_count']}")
    print("-" * 60)
    print(f"  Cleaned input preview:")
    preview = result["cleaned_input"][:200]
    print(f"  {preview}{'...' if len(result['cleaned_input']) > 200 else ''}")
    print("=" * 60 + "\n")


# =============================================================
# ── ENTRY POINT  (two-variable mode — swap these for UI later)
# =============================================================
if __name__ == "__main__":
    start_time = time.time()
    # ── USER INPUTS ─────────────────────────────────────────
    # Replace these two variables with UI inputs later.
    input_title = "আইটি ট্রেনিং সেন্টারের ভিত্তি স্থাপন করলেন রাষ্ট্রপতি"
    input_body  = """
নেত্রকোনা: নেত্রকোনা শহরের পুরাতন জেলখানা সড়কে শেখ কামাল আইটি ট্রেনিং অ্যান্ড ইনকিউবেশন সেন্টারের ভিত্তিপ্রস্তর স্থাপন করেছেন রাষ্ট্রপতি মো. আবদুল হামিদ। বুধবার বিকেলে তিনি এ ভিত্তিপ্রস্তর স্থাপন করেন। এর আগে তিনি হেলিকপ্টারে করে নেত্রকোনা বর্ডার গার্ড বাংলাদেশ (বিজিবি) ক্যাম্পে এসে পৌঁছান। অনুষ্ঠানে রাষ্ট্রপতি বলেন, এ অঞ্চলের তরুণ সমাজকে দক্ষ মানবসম্পদ হিসেবে গড়ে তুলতে এই ট্রেনিং সেন্টার গুরুত্বপূর্ণ ভূমিকা রাখবে।
    """

    # ── MODEL CHOICE ─────────────────────────────────────────
    # "bert"  → BanglaBERT   (CPU-based inference)
    # "mamba" → Bangla-Mamba (CPU: needs mamba_1024_hf/ folder)
    model_choice = "mamba"   # ← change to "mamba" when ready
    # ─────────────────────────────────────────────────────────

    result = predict_news(
        title        = input_title,
        body         = input_body,
        model_choice = model_choice,
    )
    print_result(result)
    end_time = time.time()
    
    print(f"Total time: {end_time - start_time:.2f} seconds")

"""
=============================================================
model.py — Bangla-Mamba Sequence Classification
=============================================================
Architecture : Mamba SSM backbone + classification head
Parameters   : ~43–48M depending on vocab size
Input        : headline [SEP] content  (max 768 tokens)
Output       : 2-class logits  [Fake=0, Real=1]
=============================================================
Install dependency before importing:
  pip install mamba-ssm causal-conv1d
=============================================================
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


# ─────────────────────────────────────────────
# MODEL CONFIG DATACLASS
# ─────────────────────────────────────────────
@dataclass
class BanglaMambaConfig:
    """
    All architectural hyperparameters in one place.
    Matched to banglabert vocab size (32,000).

    Parameter budget breakdown:
      Embedding layer : 32000 × 512          = 16.4M
      Mamba blocks    : 16 × ~1.7M each      = 27.2M
      Classification  : 512 → 128 → 2        = 0.07M
      Total                                  ≈ 43.7M
    """
    # ── SSM backbone ───────────────────────────
    d_model    : int   = 512      # hidden dimension
    n_layer    : int   = 16       # number of Mamba blocks
    vocab_size : int   = 32000    # must match your tokenizer exactly
    ssm_cfg    : dict  = None     # extra SSM options (leave None for defaults)

    # ── Padding ────────────────────────────────
    pad_vocab_size_multiple: int = 8   # pads vocab to nearest multiple of 8
                                        # for faster CUDA matmul

    # ── Classification head ────────────────────
    num_labels       : int   = 2
    head_hidden_dim  : int   = 128
    head_dropout     : float = 0.2    # slightly higher than BERT to reduce
                                       # overfitting (BERT showed 99% train acc)

    # ── Pooling ────────────────────────────────
    pooling : str = "mean"   # "mean" = average all non-pad positions
                              # "last" = final token only (weaker for cls)
                              # "mean" is standard for sequence classification

    def __post_init__(self):
        if self.ssm_cfg is None:
            self.ssm_cfg = {}


# ─────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    Two-layer MLP on top of pooled Mamba hidden states.

    Input  : (batch, d_model)  — pooled sequence representation
    Output : (batch, num_labels)  — raw logits

    Design choices:
      - LayerNorm first  : stabilizes pooled features before projection
      - GELU activation  : smoother gradient than ReLU
      - Dropout (0.2)    : higher than BERT (0.1) to counter overfitting
      - Two linear layers: adds capacity vs direct 512→2 projection
    """
    def __init__(self, cfg: BanglaMambaConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────
class BanglaMambaForClassification(nn.Module):
    """
    Mamba SSM backbone with a mean-pooled classification head.

    Forward pass:
      1. input_ids  → Mamba backbone → hidden states (B, L, D)
      2. hidden states × attention_mask → mean pool → (B, D)
      3. pooled → classification head → logits (B, 2)

    Why mean pooling over last-token:
      Mamba is causal (left-to-right). The last token aggregates
      all context BUT in long articles (768 tokens) this single
      position may not capture early headline signals as well
      as averaging across all positions.
    """

    def __init__(self, cfg: BanglaMambaConfig):
        super().__init__()
        self.cfg = cfg

        # ── Build Mamba backbone ──────────────────
        mamba_config = MambaConfig(
            d_model                = cfg.d_model,
            n_layer                = cfg.n_layer,
            vocab_size             = cfg.vocab_size,
            ssm_cfg                = cfg.ssm_cfg,
            pad_vocab_size_multiple= cfg.pad_vocab_size_multiple,
        )

        # MambaLMHeadModel contains:
        #   .backbone  → MambaModel  (embedding + SSM blocks)
        #   .lm_head   → linear projection to vocab (we discard this)
        _lm_model = MambaLMHeadModel(mamba_config)
        self.backbone = _lm_model.backbone   # MambaModel only
        del _lm_model                         # free the LM head memory

        # ── Classification head ───────────────────
        self.head = ClassificationHead(cfg)

    def forward(
        self,
        input_ids      : torch.Tensor,           # (B, L)
        attention_mask : torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Returns:
            logits : (B, num_labels)  — NOT softmaxed
        """
        # ── 1. Mamba forward ─────────────────────
        # MambaModel.forward() returns a plain tensor of shape (B, L, D),
        # NOT a dataclass — do NOT call .last_hidden_state on it.
        hidden = self.backbone(input_ids)   # (B, L, D)

        # ── 2. Mean pooling over non-pad positions ─
        if attention_mask is not None:
            # mask shape: (B, L) → (B, L, 1) for broadcasting
            mask    = attention_mask.unsqueeze(-1).to(hidden.dtype)
            # zero out pad positions, sum, divide by token count
            summed  = (hidden * mask).sum(dim=1)              # (B, D)
            count   = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
            pooled  = summed / count                           # (B, D)
        else:
            # No mask provided — plain mean over all positions
            pooled  = hidden.mean(dim=1)                       # (B, D)

        # ── 3. Classification head ────────────────
        logits = self.head(pooled)                             # (B, 2)
        return logits

    # ── Convenience methods ───────────────────────
    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        backbone  = sum(p.numel() for p in self.backbone.parameters())
        head      = sum(p.numel() for p in self.head.parameters())
        return {
            "total_M"    : round(total    / 1e6, 2),
            "backbone_M" : round(backbone / 1e6, 2),
            "head_M"     : round(head     / 1e6, 2),
        }

    def save(self, path: str):
        """Save model weights + config dict."""
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model_weights.pt")
        config_dict = {
            "d_model"         : self.cfg.d_model,
            "n_layer"         : self.cfg.n_layer,
            "vocab_size"      : self.cfg.vocab_size,
            "num_labels"      : self.cfg.num_labels,
            "head_hidden_dim" : self.cfg.head_hidden_dim,
            "head_dropout"    : self.cfg.head_dropout,
            "pooling"         : self.cfg.pooling,
        }
        with open(f"{path}/mamba_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from saved weights + config."""
        import json
        with open(f"{path}/mamba_config.json") as f:
            config_dict = json.load(f)
        cfg   = BanglaMambaConfig(**config_dict)
        model = cls(cfg)
        state = torch.load(
            f"{path}/model_weights.pt",
            map_location=device
        )
        model.load_state_dict(state)
        return model.to(device)


# ─────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────
def build_bangla_mamba(
    vocab_size: int = 32000,
    d_model: int = 512,
    n_layer: int = 16,
    num_labels: int = 2,
    head_hidden_dim: int = 128,
    head_dropout: float = 0.2,
    pooling: str = "mean",
) -> BanglaMambaForClassification:
    """
    Build the Bangla-Mamba model dynamically from custom configurations.
    """
    cfg = BanglaMambaConfig(
        d_model   = d_model,
        n_layer   = n_layer,
        vocab_size= vocab_size,
        num_labels= num_labels,
        head_hidden_dim = head_hidden_dim,
        head_dropout    = head_dropout,
        pooling         = pooling,
    )
    return BanglaMambaForClassification(cfg)
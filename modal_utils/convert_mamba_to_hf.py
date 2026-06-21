"""
=============================================================
convert_mamba_to_hf.py  — Modal Cloud Conversion Script
=============================================================
Runs on Modal (GPU) where mamba-ssm is available.
Converts the trained Bangla-Mamba weights from mamba-ssm
format → HuggingFace MambaModel format so the model can be
loaded locally on CPU without any CUDA dependencies.

Output saved to Modal volume:
    /root/datasets/Artifacts/best_model/mamba_768_hf/

Download that folder manually after running this script,
then use MambaHFPredictor in predict.py for CPU inference.

Run with:
    modal run modal_utils/convert_mamba_to_hf.py
=============================================================
"""

import modal
import sys
import os

app = modal.App("mamba-hf-convert")

volume = modal.Volume.from_name("datasets-volume")

# Same image as modal_run.py — needs mamba-ssm on GPU to load trained weights
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    # Everything in ONE run_commands call — guarantees correct order,
    .run_commands(
        # System dependencies (needed for CUDA extensions)
        "apt-get update -y && apt-get install -y "
        "g++ build-essential git",

        # Core Python build tools
        "pip install --upgrade pip setuptools wheel packaging ninja",

        # Scientific stack
        "pip install "
        "pandas numpy scikit-learn matplotlib datasets tokenizers mlflow pydantic-settings",

        # PyTorch (CUDA 12.4 build)
        "pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124",

        # Transformers pinned for mamba-ssm compatibility
        "pip install transformers==4.40.0",

        # CUDA extension dependencies (order matters)
        "pip install causal-conv1d==1.4.0 --no-build-isolation",
        "pip install mamba-ssm==2.2.2 --no-build-isolation",
    )
    .add_local_dir("src",    remote_path="/app/src")
    .add_local_dir("config", remote_path="/app/config")
)


@app.function(
    image=image,
    gpu="A100-40GB",      # GPU needed only to load mamba-ssm weights
    cpu=4,
    memory=16384,
    volumes={"/root/datasets": volume},
    timeout=60 * 30,
)
def convert():
    """
    Loads the mamba-ssm trained model from the Modal volume,
    remaps state dict keys to match HuggingFace MambaModel,
    then saves the converted model + tokenizer to the volume.
    """
    import json
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer, MambaConfig, MambaModel

    sys.path.insert(0, "/app")
    from src.ssm_model import BanglaMambaForClassification

    # ── Paths ────────────────────────────────────────────────
    BASE_DIR        = Path("/root/datasets")
    SRC_DIR         = BASE_DIR / "Artifacts" / "best_model" / "mamba_768"
    DST_DIR         = BASE_DIR / "Artifacts" / "best_model" / "mamba_768_hf"
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Source (mamba-ssm) : {SRC_DIR}")
    print(f"  Destination (HF)   : {DST_DIR}")
    print(f"{'='*60}\n")

    # ── 1. Load mamba_config.json ─────────────────────────────
    with open(SRC_DIR / "mamba_config.json") as f:
        mamba_cfg = json.load(f)

    print(f"  Loaded mamba_config.json: {mamba_cfg}\n")

    # ── 2. Load full state dict from mamba-ssm model ─────────
    print("  Loading mamba-ssm model weights...")
    state_dict = torch.load(
        SRC_DIR / "model_weights.pt",
        map_location="cpu",
    )
    print(f"  State dict keys ({len(state_dict)} total):")
    for k in list(state_dict.keys())[:10]:
        print(f"    {k}")
    print(f"    ...")

    # ── 3. Remap backbone keys: mamba-ssm → HuggingFace ──────
    #
    #  Key differences between mamba-ssm MambaModel and HF MambaModel:
    #    mamba-ssm                HuggingFace
    #    ─────────────────────    ─────────────────────────────
    #    backbone.embedding.*  →  backbone.embeddings.*
    #    (everything else is identical in name)
    #
    print("\n  Remapping state dict keys...")
    remapped = {}
    n_remapped = 0
    for key, val in state_dict.items():
        new_key = key
        if key.startswith("backbone.embedding."):
            new_key = key.replace("backbone.embedding.", "backbone.embeddings.", 1)
            n_remapped += 1
        remapped[new_key] = val

    print(f"  Keys remapped     : {n_remapped}")
    print(f"  Total keys        : {len(remapped)}")

    # ── 4. Build a HF-backed model with same architecture ─────
    #
    #  We create a thin wrapper:
    #    backbone  → transformers.MambaModel  (CPU-safe)
    #    head      → our ClassificationHead   (unchanged)
    #
    print("\n  Building HF MambaConfig...")
    hf_config = MambaConfig(
        vocab_size         = mamba_cfg["vocab_size"],
        hidden_size        = mamba_cfg["d_model"],       # d_model
        num_hidden_layers  = mamba_cfg["n_layer"],       # n_layer
        # SSM defaults (must match mamba-ssm ssm_cfg defaults)
        state_size         = 16,   # d_state default in mamba-ssm
        expand             = 2,    # expand factor default
        conv_kernel        = 4,    # d_conv default
        use_bias           = False,
        use_conv_bias      = True,
        # pad vocab to multiple of 8 (matches pad_vocab_size_multiple=8)
        pad_vocab_size_multiple = 8,
    )
    print(f"  HF MambaConfig    : hidden={hf_config.hidden_size}, "
          f"layers={hf_config.num_hidden_layers}, "
          f"vocab={hf_config.vocab_size}")

    # ── 5. Verify key compatibility ───────────────────────────
    print("\n  Verifying key compatibility against HF MambaModel...")
    hf_backbone   = MambaModel(hf_config)
    hf_state_keys = set(f"backbone.{k}" for k in hf_backbone.state_dict().keys())
    our_bb_keys   = set(k for k in remapped.keys() if k.startswith("backbone."))
    head_keys     = set(k for k in remapped.keys() if k.startswith("head."))

    missing   = hf_state_keys - our_bb_keys
    extra     = our_bb_keys - hf_state_keys

    if missing:
        print(f"  ⚠️  Missing keys in our weights : {missing}")
    if extra:
        print(f"  ⚠️  Extra keys in our weights   : {extra}")
    if not missing and not extra:
        print(f"  ✅  All backbone keys match perfectly!")
    print(f"  Head keys         : {len(head_keys)}")

    # ── 6. Save remapped weights (.pt) ────────────────────────
    print("\n  Saving remapped model_weights.pt...")
    torch.save(remapped, DST_DIR / "model_weights.pt")

    # ── 7. Save HF MambaConfig as JSON ────────────────────────
    hf_config.save_pretrained(str(DST_DIR))
    print("  Saved HF config.json")

    # ── 8. Save our custom mamba_config.json (for head params) ─
    with open(DST_DIR / "mamba_config.json", "w") as f:
        json.dump(mamba_cfg, f, indent=2)
    print("  Saved mamba_config.json (head params)")

    # ── 9. Copy tokenizer from source ─────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(SRC_DIR))
    tokenizer.save_pretrained(str(DST_DIR))
    print("  Copied tokenizer")

    # ── 10. Commit to Modal volume ────────────────────────────
    volume.commit()

    print(f"\n{'='*60}")
    print(f"  ✅  Conversion complete!")
    print(f"  HF model saved to: {DST_DIR}")
    print(f"{'='*60}")
    print("\n  Download the folder:")
    print(f"    {DST_DIR}")
    print("  Then place it at:")
    print("    Artifacts/best_model/mamba_768_hf/")
    print("  and use MambaHFPredictor in predict.py\n")


@app.local_entrypoint()
def main():
    convert.remote()

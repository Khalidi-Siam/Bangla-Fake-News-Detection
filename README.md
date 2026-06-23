# 🇧🇩 Bangla Fake News Detection

A comparative study of **BanglaBERT** (Transformer) vs **Bangla-Mamba** (State-Space Model) for binary fake news classification in the Bengali language. This project explores whether the longer context window of Mamba (768 tokens) provides a measurable advantage over the architecturally limited 512-token window of BanglaBERT — especially on longer news articles.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Cloud Training (Modal)](#cloud-training-modal)
- [CPU Inference — Mamba Conversion](#cpu-inference--mamba-conversion)
- [Streamlit App](#streamlit-app)
- [Experiment Tracking (MLflow + DagsHub)](#experiment-tracking-mlflow--dagshub)
- [Known Issues & Notes](#known-issues--notes)
- [Results](#results)

---

## Overview

### Selected Domain

This project operates in the domain of **low-resource Bangla natural language processing**, specifically targeting the automated detection of misinformation in online Bangla news articles. Bangla is the seventh most spoken language in the world with over 230 million native speakers, yet it remains severely underserved in NLP research compared to English and other high-resource languages. The rapid proliferation of Bangla digital news platforms and social media has made the spread of fake news a pressing societal concern — one that existing tools are ill-equipped to address at scale.

### Problem Statement

Existing Bangla fake news classifiers rely primarily on transformer-based models such as BanglaBERT, which inherit a fundamental architectural limitation: a fixed attention window of **512 tokens**. In practice, this forces the model to silently truncate any article longer than roughly 300-400 words — discarding up to half the content of a typical Bangla news article before classification even begins.

This truncation is not a minor inconvenience. It is not necessary that fake news articles always exhibit their most revealing inconsistencies in the headline or opening paragraphs. In many cases, signals such as contradictory claims, fabricated citations, or misleading context may appear in the middle or toward the end of an article. A model that cannot process the full article is therefore structurally blind to these important indicators, limiting its ability to reliably detect misinformation.

Furthermore, transformer attention scales quadratically with sequence length — making it computationally prohibitive to extend the context window beyond 512 tokens without dramatic increases in memory and inference time. This project investigates whether **State Space Models (SSMs)**, specifically the Mamba architecture, can address both limitations simultaneously: providing a longer effective context window with linear computational complexity, trained entirely from scratch without reliance on large pretraining corpora.

Specifically, this project fine-tunes **BanglaBERT** (`csebuetnlp/banglabert`) on the **BanFakeNews-2.0** dataset and simultaneously trains a **Bangla-Mamba** model (Mamba SSM backbone, ~43.7M parameters) from scratch using the same tokenizer — enabling a fair architectural comparison.

**Core thesis question:** Does Mamba's larger context window (768 vs 512 tokens) give it a meaningful advantage on *long* Bangla news articles that BanglaBERT must truncate?

The test set is intentionally split into two subsets:
- **Short** (≤ 512 tokens): Both models see the full text — a fair baseline.
- **Long** (> 512 tokens): BanglaBERT truncates; Mamba reads everything.

### Expected Output

| # | Output | Description |
|---|--------|-------------|
| 1 | Binary classification model | A trained Bangla-Mamba-43M model that classifies any Bangla news article (headline + body) as authentic or fake, accepting inputs up to 768 tokens |
| 2 | Empirical long-context advantage | Quantitative evidence that Mamba's performance gap versus BanglaBERT narrows significantly on articles exceeding 512 tokens — where transformer truncation is most damaging |
| 3 | Efficiency benchmark | Demonstration that a 43.5M parameter SSM trained from scratch achieves similar to BanglaBERT's performance with 39% of the parameters and zero pretraining data |
| 4 | Reproducible MLOps pipeline | A full pipeline covering preprocessing, tokenization caching, model training, evaluation, and experiment tracking via MLflow on DagsHub and runnable on Modal cloud. |

---

## Models

### BanglaBERT (Baseline)
| Property | Value |
|---|---|
| Base model | `csebuetnlp/banglabert` |
| Architecture | Transformer encoder (BERT-style) |
| Max context | 512 tokens |
| Input format | `headline [SEP] content` |
| Task | Binary classification (Fake=0, Real=1) |
| Training | Full fine-tuning |
| Precision | BF16 (A100) |
| Epochs | 5 |
| Optimizer | AdamW + OneCycleLR |
| Effective batch | 64 (32 × 2 grad accum) |

### Bangla-Mamba (Proposed)
| Property | Value |
|---|---|
| Architecture | Mamba SSM backbone + 2-layer MLP head |
| Parameters | ~43.7M |
| Max context | **768 tokens** (vs BERT's 512) |
| Hidden dim | 512 |
| Mamba blocks | 16 |
| Vocab | 32,000 (BanglaBERT tokenizer) |
| Pooling | Mean over non-pad positions |
| Training | From scratch |
| Precision | BF16 (A100) |
| Learning rate | 1e-3 (higher than BERT — trained from scratch) |

The Mamba backbone uses `mamba-ssm`'s `MambaLMHeadModel` with the LM head discarded, keeping only the encoder backbone. The classification head is a `LayerNorm → Linear(512→128) → GELU → Dropout(0.2) → Linear(128→2)` MLP.

---

## Dataset

**BanFakeNews-2.0** — available on [HuggingFace](https://huggingface.co/datasets/hrshihab/BanFakeNews-2.0)

| Split | Total |
|---|---|
| Train | ~80% |
| Validation | ~10% |
| Test | ~10% |

**Label distribution (after cleaning):**
- Real (1): ~83.5% of articles (majority class)
- Fake (0): ~16.5% of articles (minority class)

Class imbalance is addressed via **weighted cross-entropy loss** with computed class weights: `Fake=3.0295, Real=0.5988`.

**Preprocessing:**
- Merge headline and content as: `headline [SEP] content`
- Strip HTML tags and URLs
- NFC Unicode normalization (fixes overlapping Bangla characters)
- Collapse whitespace, preserve punctuation (! ? ... are fake-news signals)
- Filter: minimum 20 words, maximum 2000 words
- Remove empty strings and duplicates

---

## Project Structure

```
Bangla-Fake-News-Detection/
│
├── app/
│   └── app.py                  # Streamlit web app (demo UI)
│
├── config/
│   ├── config.py               # Pydantic settings — file paths & MLflow URIs
│   └── params.py               # Pydantic settings — training hyperparameters
│
├── modal_utils/
│   ├── convert_mamba_to_hf.py  # Converts mamba-ssm weights → HuggingFace format (GPU/Modal)
│   ├── upload_folder.py        # Upload artifacts to Modal volume
│   ├── download_folder.py      # Download artifacts from Modal volume
│   ├── upload_file.py          # Upload single file to Modal volume
│   ├── delete_file.py          # Delete file from Modal volume
│   └── delete_folder.py        # Delete folder from Modal volume
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── data_ingestion.py       # Download, merge, clean, save dataset
│   ├── offline_tokenize.py     # Tokenize & cache dataset to disk (HuggingFace datasets)
│   ├── finetune_bert.py        # BanglaBERT fine-tuning pipeline
│   ├── evaluate_bert.py        # BanglaBERT evaluation (test set + thesis subsets)
│   ├── ssm_model.py            # Bangla-Mamba model architecture
│   ├── ssm_train.py            # Bangla-Mamba training pipeline
│   ├── evaluate_ssm.py         # Bangla-Mamba evaluation (test set + thesis subsets)
│   ├── predict.py              # Inference — BertPredictor / MambaPredictor
│   └── utils/
│       ├── common.py           # Shared utilities (create_directory, save_json, section)
│       ├── exception.py        # Custom exception with file/line info
│       └── logger.py           # Centralized logging setup
│
├── Artifacts/                  # Auto-generated — model weights, caches, logs
├── logs/                       # Pipeline run logs
│
├── main.py                     # Pipeline entry point (uncomment stages to run)
├── modal_run.py                # Modal cloud runner (GPU training)
├── Dockerfile                  # Docker image for Streamlit app
├── requirements.txt            # Python dependencies
└── .env                        # (not committed) MLflow credentials
```

---

## Pipeline Stages

The pipeline is orchestrated through `main.py`. Each stage is independent — run them in order by uncommenting the relevant block.

```
Stage 1: Data Ingestion        → Artifacts/data.csv
Stage 2: Offline Tokenization  → Artifacts/tokenized_cache_bert/  (or mamba)
Stage 3: BanglaBERT Fine-Tuning → Artifacts/best_model/banglabert/
Stage 4: BanglaBERT Evaluation  → Artifacts/logs/banglabert_results.json
Stage 5: Bangla-Mamba Training  → Artifacts/best_model/mamba_768/
Stage 6: Bangla-Mamba Evaluation → Artifacts/logs/mamba_768_results.json
```

**Tokenization note:** If `max_length=512`, the same tokenized cache can be shared between BanglaBERT and Mamba. For Mamba at `max_length=768`, a separate cache must be generated. Set `max_length` in `config/config.py` before running the tokenization stage.

---

## Setup & Installation

### Prerequisites
- Python 3.11
- GPU with CUDA 12.4+ (required for training; inference works on CPU)
- [Modal](https://modal.com/) account (optional — for cloud GPU training)

### Local Setup

```bash
git clone https://github.com/Khalidi-Siam/Bangla-Fake-News-Detection.git
cd Bangla-Fake-News-Detection

# Create and activate a virtual environment
python -m venv env
.\env\Scripts\activate   # Windows
# source env/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### PyTorch Version Notes

> ⚠️ **Windows users:** `torch==2.4.0` has known issues on Windows. Use `torch==2.12.1` instead.
>
> **Docker / Linux (GPU):** `torch==2.4.0` works correctly for both CPU and GPU builds.

```bash
# CPU-only (default in requirements.txt)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.4) — replace the line in requirements.txt
pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### GPU-only Dependencies (Mamba)

`causal-conv1d` and `mamba-ssm` **require CUDA** and cannot be installed on CPU-only machines. They are commented out in `requirements.txt` by default.

```bash
# Only install if you have a CUDA GPU
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2 --no-build-isolation
```

If you do not have a GPU, see [CPU Inference — Mamba Conversion](#cpu-inference--mamba-conversion) to use the converted HuggingFace format instead.

### Environment Variables

Create a `.env` file in the project root for MLflow / DagsHub credentials:

```env
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

---

## Running the Pipeline

Edit `main.py` and uncomment the stage(s) you want to run, then execute:

```bash
python main.py
```

### Stage-by-Stage

**Stage 1 — Data Ingestion**
```python
# In main.py, uncomment:
stage = "Data Ingestion"
data_ingestion = DataIngestion()
data_ingestion.initialize_data_ingestion()
```

**Stage 2 — Offline Tokenization** *(set `max_length` in config first)*
```python
stage = "Offline Tokenization"
offline_tokenizer = OfflineTokenize()
offline_tokenizer.initialize_tokenization()
```

**Stage 3 — BanglaBERT Fine-Tuning** *(requires GPU; auto-triggers evaluation)*
```python
stage = "BanglaBERT Fine-Tuning"
bert_finetuner = BertFineTune()
bert_finetuner.initialize_bert_finetuning()
```

**Stage 5 — Bangla-Mamba Training** *(requires GPU with mamba-ssm installed)*
```python
stage = "Bangla-Mamba Training"
mamba_trainer = MambaTrainer()
mamba_trainer.initialize_mamba_training()
```

---

## Cloud Training (Modal)

Training is designed to run on [Modal](https://modal.com/) using an **NVIDIA A100-40GB** GPU. The `modal_run.py` script provisions a CUDA 12.4 Docker image, installs all dependencies (including `mamba-ssm`), and runs `main.py` remotely.

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Run training on Modal cloud (A100-40GB, 4-hour timeout)
modal run modal_run.py
```

The Modal runner uses a persistent volume (`datasets-volume`) to store artifacts between runs. Utility scripts in `modal_utils/` allow you to upload/download files from the volume.

**Expected runtime on A100-40GB:**
- BanglaBERT fine-tuning: ~3 min/epoch → ~15 min total (5 epochs, BF16, batch=32)
- Bangla-Mamba training: varies (trained from scratch, higher LR)

---

## CPU Inference — Mamba Conversion

`causal-conv1d` and `mamba-ssm` require CUDA kernel compilation and **cannot run on CPU**. To use the trained Mamba model locally (without a GPU), you must convert it to the HuggingFace `MambaModel` format:

### Step 1: Convert on Modal (requires GPU)

```bash
modal run modal_utils/convert_mamba_to_hf.py
```

This script:
1. Loads the trained weights from `Artifacts/best_model/mamba_768/`
2. Remaps state dict keys (`backbone.embedding.*` → `backbone.embeddings.*`)
3. Saves a HuggingFace-compatible model to `Artifacts/best_model/mamba_768_hf/`
4. Commits the result to the Modal volume

### Step 2: Download to local machine

```bash
# Using the modal_utils download helper (or modal CLI)
python modal_utils/download_folder.py
```

Place the downloaded folder at: `Artifacts/best_model/mamba_768_hf/`

### Step 3: CPU inference

The `MambaPredictor` in `src/predict.py` automatically detects CUDA availability:
- **CUDA available** → loads from `mamba_768/` using `mamba-ssm` (fast)
- **CPU only** → loads from `mamba_768_hf/` using HuggingFace `MambaModel` (pure PyTorch)

---

## Streamlit App

An interactive web demo is included at `app/app.py`. It supports both BanglaBERT and Bangla-Mamba inference with a bilingual (Bengali + English) UI.

### Run locally

```bash
streamlit run app/app.py
```

### Run with Docker

```bash
docker build -t bangla-fake-news .
docker run -p 8501:8501 bangla-fake-news
```

Then open [http://localhost:8501](http://localhost:8501).

**App features:**
- Model selector: BanglaBERT or Bangla-Mamba
- Bilingual UI (Bengali input fields + English labels)
- Quick-load example articles (2 real, 2 fake)
- Probability breakdown meter (Fake / Real)
- Inference latency, token count, backend metadata
- Preprocessing inspector (cleaned input text)

> ⚠️ The app requires pre-trained model weights to be present in `Artifacts/best_model/`. Both models run on **CPU** in the app.

---

## Experiment Tracking (MLflow + DagsHub)

All training runs are tracked via MLflow on [DagsHub](https://dagshub.com/Khalidi-Siam/Bangla-Fake-News-Detection).

**Tracking URI:** `https://dagshub.com/Khalidi-Siam/Bangla-Fake-News-Detection.mlflow`

**Logged per run:**
- All hyperparameters (model name, batch size, LR, max_length, etc.)
- Per-epoch: train loss, train accuracy, val Macro-F1, val AUC-ROC
- Final test metrics: Macro-F1, AUC-ROC, per-class precision/recall/F1
- Model artifacts (optional — set `log_model: False` in config to skip ~500 MB upload)

Configure credentials via `.env` (see [Setup](#setup--installation)).

---

## Known Issues & Notes

### PyTorch on Windows
`torch==2.4.0` causes errors on Windows. Use `torch==2.12.1` if running natively on Windows. The Docker image (Linux) works fine with `2.4.0`.

### Mamba Requires GPU to Train
`causal-conv1d` and `mamba-ssm` are CUDA extensions that require a GPU to install and run. There is no CPU fallback for training. Use Modal for cloud GPU access, then convert the trained weights to HuggingFace format for local CPU inference (see [CPU Inference](#cpu-inference--mamba-conversion)).

### Shared Tokenization Cache
Both BanglaBERT and Mamba use the same `csebuetnlp/banglabert` tokenizer. If you use `max_length=512`, the same tokenizer cache can be reused for both models. For Mamba at `max_length=768`, run tokenization separately and update the `cache_dir` in `config/config.py`.

### Class Imbalance
The dataset is heavily imbalanced (~83.5% Real, ~16.5% Fake). Weighted cross-entropy loss is applied with pre-computed class weights (`Fake=3.03, Real=0.60`). The Macro-F1 score (not accuracy) is used as the primary evaluation metric to account for this imbalance.

### Mamba Model State Dict Keys
The `mamba-ssm` library names the embedding layer `backbone.embedding.*`, while HuggingFace `MambaModel` uses `backbone.embeddings.*`. The conversion script (`convert_mamba_to_hf.py`) handles this remapping automatically.

### BF16 Support
BF16 mixed precision is used for training on A100 GPUs (no `GradScaler` needed). On GPUs without BF16 support, the pipeline automatically falls back to FP32.

### Checkpoint Auto-Resume
Both training pipelines (`finetune_bert.py`, `ssm_train.py`) automatically detect and resume from the latest saved checkpoint if training is interrupted. Only the 2 most recent checkpoints are retained to save disk space.

---

## Results

### Key Results

| Metric | BanglaMamba-43M | BanglaBERT-110M |
|--------|-----------------|----------------|
| Full test Macro-F1 | 0.8979 | 0.9457 |
| Short article Macro-F1 (<512 tokens) | **0.8987** | 0.9544 |
| Long article Macro-F1 (>512 tokens) | **0.8943** | 0.9070 |
| Performance gap on long articles | 0.44% | **4.74%** |
| Parameters | **43.5M** | 110M |
| Pretraining | **Only on Bangla Fake News Dataset** | Large Bangla corpus |
| Training time (A100-40GB) | **approx. 22 min** | approx. 20 min (fine-tune only) |
| VRAM usage (inference) | **approx. 1.55 GB** | approx. 4.14 GB |

The evaluation includes:
- Overall test set: Accuracy, Macro-F1, AUC-ROC, per-class Precision/Recall/F1
- **Short test subset** (≤ 512 tokens): both models see full text — baseline comparison
- **Long test subset** (> 512 tokens): BanglaBERT truncates, Mamba reads everything — key thesis experiment

---

"""
=============================================================
Bangla-Mamba Fake News Classification
Offline Tokenization & Caching Pipeline
=============================================================
Tokenizer: csebuetnlp/banglabert
Context  : 768 tokens
=============================================================
"""

import os
import gc
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")
from datasets import load_from_disk
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import section
from config.config import settings


class OfflineTokenize:
    """
    Offline Tokenization pipeline: load cleaned CSV → load tokenizer →
    compute token lengths → split → compute class weights → tokenize & cache → sanity check.
    """
    # ── Tokenizer ──────────────────────────────
    TOKENIZER_NAME    = "csebuetnlp/banglabert"
    # ── Split ratios ───────────────────────────
    TRAIN_RATIO       = 0.80
    VAL_RATIO         = 0.10
    TEST_RATIO        = 0.10

    def __init__(self):
        self.cache_dir             = settings.offline_tokenize.cache_dir
        self.max_length            = settings.offline_tokenize.max_length
        self.cleaned_dataset_path  = settings.data.cleaned_dataset_path
        self.seed                  = settings.seed
        self.class_weight_path     = settings.offline_tokenize.class_weight_path
        self.short_test_subset_path = settings.offline_tokenize.short_test_subset_path
        self.long_test_subset_path  = settings.offline_tokenize.long_test_subset_path
        self.df                    = None
        self.train_df              = None
        self.val_df                = None
        self.test_df               = None
        self.long_test_df          = None
        self.short_test_df         = None
        self.class_weights         = None
        self.weight_tensor         = None
        self.tokenizer             = None


    # ──────────────────────────────────────────────────────────
    # STEP 1 — LOAD CLEANED DATASET
    # ──────────────────────────────────────────────────────────
    def load_cleaned_dataset(self):
        try:
            section("STEP 1 · Loading Cleaned Dataset")
            if not os.path.exists(self.cleaned_dataset_path):
                raise FileNotFoundError(f"Cleaned dataset not found at {self.cleaned_dataset_path}")

            logging.info(f"  Loading: {self.cleaned_dataset_path}")
            self.df = pd.read_csv(self.cleaned_dataset_path)
            logging.info(f"  Loaded cleaned dataset: {len(self.df):,} rows")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 2 — TOKENIZER LOAD
    # ──────────────────────────────────────────────────────────
    def load_tokenizer(self):
        try:
            section("STEP 2 · Tokenizer — Load")

            logging.info(f"  Loading: {self.TOKENIZER_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logging.info("  pad_token was None → set to eos_token")

            logging.info(f"  Vocabulary size  : {self.tokenizer.vocab_size:,}")
            logging.info(f"  Pad token        : '{self.tokenizer.pad_token}' (id={self.tokenizer.pad_token_id})")

            # ── Confirm [SEP] token exists ────────────
            sep_id    = self.tokenizer.convert_tokens_to_ids('[SEP]')
            is_native = sep_id != self.tokenizer.unk_token_id
            logging.info(f"  [SEP] token id   : {sep_id}  "
                          f"{'native token' if is_native else 'mapped to UNK — use || instead'}")

            if not is_native:
                logging.warning("  Switching separator to '||' ...")
                self.df['content'] = self.df['content'].str.replace(' [SEP] ', ' || ', regex=False)

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 3 — TOKEN LENGTH STATS ON FULL DATASET
    # ──────────────────────────────────────────────────────────
    def compute_token_lengths(self):
        try:
            section("STEP 3 · Token Length Stats on Full Dataset")

            logging.info("  Tokenizing full dataset for length statistics...")
            logging.info("  (This may take 3–8 minutes depending on CPU)")

            def get_token_length(text: str) -> int:
                return len(self.tokenizer.encode(
                    text, add_special_tokens=True,
                    max_length=2000, truncation=True
                ))

            self.df['token_count'] = self.df['content'].apply(get_token_length)

            logging.info("\n  Token count distribution (merged headline + content):")
            logging.info(self.df['token_count'].describe().round(1).to_string())

            pct_512  = (self.df['token_count'] >  512).mean() * 100
            pct_768  = (self.df['token_count'] >  768).mean() * 100
            pct_1024 = (self.df['token_count'] > 1024).mean() * 100
            pct_fit  = (self.df['token_count'] <= 512).mean() * 100

            logging.info("\n  Context window coverage:")
            logging.info(f"    Fit in BanglaBERT-512  (no truncation) : {pct_fit:.1f}%")
            logging.info(f"    Truncated by BanglaBERT (> 512 tokens) : {pct_512:.1f}%")
            logging.info(f"    Truncated by Mamba-768  (> 768 tokens) : {pct_768:.1f}%")
            logging.info(f"    Exceed 1024 tokens                     : {pct_1024:.1f}%")
            logging.info(f"\n  Thesis argument: Mamba-768 fully reads {(100-pct_768):.1f}% of all articles")
            logging.info(f"     vs BanglaBERT which fully reads only {pct_fit:.1f}%")

            logging.info("\n  Token count by label:")
            logging.info(self.df.groupby('label')['token_count'].describe().round(1).to_string())

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 4 — STRATIFIED TRAIN / VAL / TEST SPLIT
    # ──────────────────────────────────────────────────────────
    def split(self):
        try:
            section("STEP 4 · Stratified Train / Val / Test Split  (80 / 10 / 10)")

            self.train_df, temp_df = train_test_split(
                self.df,
                test_size    = 1 - self.TRAIN_RATIO,
                stratify     = self.df['label'],
                random_state = self.seed
            )
            self.val_df, self.test_df = train_test_split(
                temp_df,
                test_size    = self.TEST_RATIO / (self.VAL_RATIO + self.TEST_RATIO),
                stratify     = temp_df['label'],
                random_state = self.seed
            )

            for name, split_data in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
                total = len(split_data)
                fake  = (split_data['label'] == 0).sum()
                real  = (split_data['label'] == 1).sum()
                logging.info(f"  {name:5s}: {total:6,} rows  |  "
                              f"Fake={fake:,} ({fake/total*100:.1f}%)  "
                              f"Real={real:,} ({real/total*100:.1f}%)")

            self.train_df = self.train_df.reset_index(drop=True)
            self.val_df   = self.val_df.reset_index(drop=True)
            self.test_df  = self.test_df.reset_index(drop=True)

            # ── Long / short test subsets for thesis experiment ──
            self.long_test_df  = self.test_df[self.test_df['token_count'] >  512].reset_index(drop=True)
            self.short_test_df = self.test_df[self.test_df['token_count'] <= 512].reset_index(drop=True)

            logging.info("\n  Thesis experiment subsets (from test set):")
            logging.info(f"    Long  (> 512 tokens) : {len(self.long_test_df):,} articles  "
                          "← BanglaBERT truncates these")
            logging.info(f"    Short (≤ 512 tokens) : {len(self.short_test_df):,} articles  "
                          "<= both models see full text")

            self.long_test_df.to_csv(self.long_test_subset_path,   index=False)
            self.short_test_df.to_csv(self.short_test_subset_path,  index=False)
            logging.info(f"  Saved → {self.long_test_subset_path}")
            logging.info(f"  Saved → {self.short_test_subset_path}")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 5 — CLASS WEIGHT CALCULATION
    # ──────────────────────────────────────────────────────────
    def compute_class_weights(self):
        try:
            section("STEP 5 · Class Weight Calculation")

            self.class_weights = compute_class_weight(
                class_weight = 'balanced',
                classes      = np.array([0, 1]),
                y            = self.train_df['label'].values
            )
            self.weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32)

            logging.info("  Class weights (balanced):")
            logging.info(f"    Fake (0) : {self.class_weights[0]:.4f}  ← upweighted (minority class)")
            logging.info(f"    Real (1) : {self.class_weights[1]:.4f}  ← downweighted (majority class)")
            logging.info("\n  In your training script use:")
            logging.info(f"    criterion = nn.CrossEntropyLoss(weight=torch.tensor({self.weight_tensor.tolist()}).to(device))")

            np.save(self.class_weight_path, self.class_weights)
            logging.info(f"\n  Saved → {self.class_weight_path}")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 6 — OFFLINE TOKENIZATION & DISK CACHING
    # ──────────────────────────────────────────────────────────
    def tokenize_and_cache(self):
        try:
            section("STEP 6 · Offline Tokenization & Disk Caching")

            os.makedirs(self.cache_dir, exist_ok=True)

            def gen_split_data(dataframe):
                for _, row in dataframe.iterrows():
                    yield {
                        "content": str(row['content']),
                        "label": int(row['label'])
                    }

            def tokenize_batch(examples):
                return self.tokenizer(
                    examples['content'],
                    max_length     = self.max_length,
                    truncation     = True,
                    padding        = 'max_length',
                    return_tensors = None,
                )

            splits = {
                'train' : self.train_df[['content', 'label']],
                'val'   : self.val_df[['content', 'label']],
                'test'  : self.test_df[['content', 'label']],
            }

            for split_name, split_data in splits.items():
                logging.info(f"\n  Streaming and Tokenizing {split_name} ({len(split_data):,} articles)...")

                hf_ds = Dataset.from_generator(
                    gen_split_data,
                    gen_kwargs={"dataframe": split_data}
                )

                tok_ds = hf_ds.map(
                    tokenize_batch,
                    batched           = True,
                    batch_size        = 256,
                    desc              = f"  Tokenizing {split_name}",
                    remove_columns    = ['content'],
                    writer_batch_size = 1000,
                    keep_in_memory    = False
                )

                tok_ds.set_format(
                    type    = 'torch',
                    columns = ['input_ids', 'attention_mask', 'label']
                )

                save_path = f"{self.cache_dir}/{split_name}"
                tok_ds.save_to_disk(save_path)
                logging.info(f"  Saved → {save_path}")

                del hf_ds, tok_ds
                gc.collect()

            logging.info(f"\n  Cache directory contents:")
            for p in sorted(Path(self.cache_dir).iterdir()):
                logging.info(f"    {p}")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 7 — SANITY CHECK
    # ──────────────────────────────────────────────────────────
    def sanity_check(self):
        try:
            section("STEP 7 · Sanity Check on Cached Data")

            for split_name in ['train', 'val', 'test']:
                ds     = load_from_disk(f"{self.cache_dir}/{split_name}")
                sample = ds[0]

                assert sample['input_ids'].shape     == torch.Size([self.max_length])
                assert sample['attention_mask'].shape == torch.Size([self.max_length])

                non_pad = sample['attention_mask'].sum().item()
                label   = sample['label'].item()

                logging.info(f"  {split_name:5s} — rows: {len(ds):,} | "
                             f"input shape: [{self.max_length}] | "
                             f"non-pad tokens: {non_pad} | "
                             f"label: {label}")

            train_ds      = load_from_disk(f"{self.cache_dir}/train")
            ids           = train_ds[0]['input_ids'].tolist()
            sep_id        = self.tokenizer.convert_tokens_to_ids('[SEP]')
            sep_positions = [i for i, x in enumerate(ids) if x == sep_id]
            logging.info(f"\n  [SEP] token (id={sep_id}) found at positions: {sep_positions[:5]}")
            if len(sep_positions) >= 2:
                logging.info("  Headline separator correctly encoded in input_ids")
            else:
                logging.warning("  Only one [SEP] found (CLS...tokens...[SEP]) — this is still fine")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 8 — FINAL SUMMARY
    # ──────────────────────────────────────────────────────────
    def print_summary(self):
        try:
            section("STEP 8 · Final Tokenization Summary — Ready for Training")

            summary = f"""
  +--------------------------------------------------+
  |          TOKENIZATION COMPLETE                   |
  +--------------------------------------------------+
  |  Train              : {len(self.train_df):>8,}                  |
  |  Val                : {len(self.val_df):>8,}                  |
  |  Test               : {len(self.test_df):>8,}                  |
  |  Long test (>512)   : {len(self.long_test_df):>8,}                  |
  |  Short test (<=512) : {len(self.short_test_df):>8,}                  |
  +--------------------------------------------------+
  |  Tokenizer          : {self.TOKENIZER_NAME}       |
  |  Vocab size         : {self.tokenizer.vocab_size:>8,}                  |
  |  Max length         : {self.max_length:>8}                  |
  +--------------------------------------------------+
  |  Class weight Fake  : {self.class_weights[0]:>8.4f}                  |
  |  Class weight Real  : {self.class_weights[1]:>8.4f}                  |
  +--------------------------------------------------+
  |  Cache dir          : {str(self.cache_dir):<25} |
  +--------------------------------------------------+

  LOAD IN YOUR TRAINING SCRIPT:
  --------------------------------------------------
  from datasets import load_from_disk
  import numpy as np, torch

  train_ds = load_from_disk('{self.cache_dir}/train')
  val_ds   = load_from_disk('{self.cache_dir}/val')
  test_ds  = load_from_disk('{self.cache_dir}/test')

  train_ds.set_format(type='torch',
                      columns=['input_ids', 'attention_mask', 'label'])

  class_weights = np.load('{self.class_weight_path}')
  weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
"""
            logging.info(summary)

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_tokenization(self):
        """Execute the tokenization pipeline end-to-end starting from cleaned CSV."""
        self.load_cleaned_dataset()
        self.load_tokenizer()
        self.compute_token_lengths()
        self.split()
        self.compute_class_weights()
        self.tokenize_and_cache()
        self.sanity_check()
        self.print_summary()

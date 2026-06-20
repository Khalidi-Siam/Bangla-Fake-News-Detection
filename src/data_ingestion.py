"""
=============================================================
Bangla-Mamba Fake News Classification
Data Download + Clean Pipeline
=============================================================
Dataset  : BanFakeNews-2.0 (Mendeley / Kaggle)
Input    : headline [SEP] content
=============================================================
"""

import os
import sys
import re
import unicodedata
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import section, create_directory
from config.config import settings


class DataIngestion:
    """
    Data Ingestion pipeline: load → merge headline → clean → save CSV.
    """
    # ── Data filters ───────────────────────────
    MIN_WORDS      = 20
    MAX_WORDS      = 2000

    def __init__(self):
        self.hf_dataset_name = settings.data.hf_dataset_name
        self.cleaned_dataset_path = settings.data.cleaned_dataset_path
        self.df = None

        

    # ──────────────────────────────────────────────────────────
    # STEP 1 — DOWNLOAD DATASET
    # ──────────────────────────────────────────────────────────
    def download_dataset(self):
        try:
            section("STEP 1 · Downloading Dataset from HuggingFace")

            logging.info(f"  Downloading: {self.hf_dataset_name}")
            self.df = load_dataset(self.hf_dataset_name)["train"].to_pandas()
            logging.info(f"  Loaded 'train' split: {len(self.df):,} rows")

            # ── Lowercase all column names ─────────────────
            self.df.columns = self.df.columns.str.lower()
            logging.info(f"  Columns    : {list(self.df.columns)}")

            # ── Validate required columns exist ────────────
            for col in ('headline', 'content', 'label'):
                if col not in self.df.columns:
                    raise ValueError(
                        f"Expected column '{col}' not found after lowercasing.\n"
                        f"Available columns: {list(self.df.columns)}"
                    )

            # ── Keep only the three columns we need ────────
            self.df = self.df[['headline', 'content', 'label']].copy()

            logging.info("\n  Raw label distribution:")
            logging.info(self.df['label'].value_counts().to_string())


        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 2 — HEADLINE + CONTENT MERGE
    # ──────────────────────────────────────────────────────────
    def merge_headline(self):
        try:
            section("STEP 2 · Headline + Content Merge")

            if 'headline' in self.df.columns:
                self.df['headline'] = self.df['headline'].fillna('').astype(str).str.strip()
                self.df['content']  = self.df['content'].fillna('').astype(str).str.strip()

                non_empty_headlines = (self.df['headline'] != '').sum()
                logging.info(f"  Articles with non-empty headline : {non_empty_headlines:,} "
                             f"({non_empty_headlines/len(self.df)*100:.1f}%)")

                # Merge: "Headline text [SEP] Article body..."
                self.df['content'] = self.df['headline'] + ' [SEP] ' + self.df['content']
                self.df = self.df.drop(columns=['headline'])

                logging.info("  Merged as: headline [SEP] content")
                logging.info(f"  Sample merged input: '{self.df['content'].iloc[0][:120]}...'")
            else:
                logging.info("  No headline column — using content only.")
                self.df['content'] = self.df['content'].fillna('').astype(str).str.strip()

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 3 — CLEANING
    # ──────────────────────────────────────────────────────────
    def clean(self):
        try:
            section("STEP 3 · Cleaning")

            def clean_bangla(text: str) -> str:
                """
                Cleans merged (headline + content) Bangla text:
                  - Strip HTML tags
                  - Remove URLs
                  - Normalize unicode to NFC  (fixes overlapping Bangla chars)
                  - Collapse whitespace
                  - Preserve all punctuation  (!, ?, ... are fake-news signals)
                  - Preserve [SEP] separator token
                """
                if not isinstance(text, str):
                    return ""
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'http\S+|www\.\S+', ' ', text)
                text = unicodedata.normalize('NFC', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            logging.info("  Applying cleaning function...")
            self.df['content'] = self.df['content'].apply(clean_bangla)

            before = len(self.df)
            self.df = self.df[self.df['content'].str.len() > 0].copy()
            logging.info(f"  Removed empty strings  : {before - len(self.df)}")

            before = len(self.df)
            self.df = self.df.drop_duplicates(subset='content').reset_index(drop=True)
            logging.info(f"  Removed duplicates     : {before - len(self.df)}")

            self.df['word_count'] = self.df['content'].apply(lambda x: len(x.split()))
            before = len(self.df)
            self.df = self.df[(self.df['word_count'] >= self.MIN_WORDS) &
                    (self.df['word_count'] <= self.MAX_WORDS)].reset_index(drop=True)
            logging.info(f"  Removed by word filter : {before - len(self.df)}  "
                         f"(kept {self.MIN_WORDS}–{self.MAX_WORDS} words)")

            self.df['label'] = self.df['label'].astype(int)
            assert set(self.df['label'].unique()).issubset({0, 1}), \
                f"Labels must be 0 or 1, found: {self.df['label'].unique()}"

            logging.info(f"\n  Dataset after cleaning : {len(self.df):,} rows")
            logging.info("  Label distribution:")
            for lbl, cnt in self.df['label'].value_counts().items():
                name = "Real (1)" if lbl == 1 else "Fake (0)"
                logging.info(f"    {name}: {cnt:,}  ({cnt/len(self.df)*100:.1f}%)")

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 4 — SAVE CLEANED CSV
    # ──────────────────────────────────────────────────────────
    def save_clean_csv(self):
        try:
            section("STEP 4 · Save Cleaned Dataset")
            create_directory(Path(self.cleaned_dataset_path).parent)
            logging.info(f"  Saving cleaned dataset to {self.cleaned_dataset_path}...")
            self.df.to_csv(self.cleaned_dataset_path, index=False)
            logging.info("  Saved successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # STEP 5 — PRINT INGESTION SUMMARY
    # ──────────────────────────────────────────────────────────
    def print_ingestion_summary(self):
        try:
            section("STEP 5 · Final Ingestion Summary")

            real_cnt = (self.df['label'] == 1).sum()
            fake_cnt = (self.df['label'] == 0).sum()

            summary = f"""
  +--------------------------------------------------+
  |            INGESTION COMPLETE                    |
  +--------------------------------------------------+
  |  Input format       : headline [SEP] content     |
  |  Total articles     : {len(self.df):>8,}                  |
  |  Real (1)           : {real_cnt:>8,}                  |
  |  Fake (0)           : {fake_cnt:>8,}                  |
  +--------------------------------------------------+
  |  Cleaned CSV path   : {str(self.cleaned_dataset_path):<25} |
  +--------------------------------------------------+
"""
            logging.info(summary)

        except Exception as e:
            raise CustomException(e, sys)

    # ──────────────────────────────────────────────────────────
    # PUBLIC: run full pipeline
    # ──────────────────────────────────────────────────────────
    def initialize_data_ingestion(self):
        """Execute the full data ingestion pipeline end-to-end."""
        self.download_dataset()
        self.merge_headline()
        self.clean()
        self.save_clean_csv()
        self.print_ingestion_summary()

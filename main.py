from src.data_ingestion import DataIngestion
from src.offline_tokenize import OfflineTokenize
from src.finetune_bert import BertFineTune
from src.evaluate_bert import BertEvaluate
from src.utils.logger import logging

# ── Data Ingestion ──
# stage = "Data Ingestion"
# logging.info(f"Stage {stage} started")
# data_ingestion = DataIngestion()
# data_ingestion.initialize_data_ingestion()


# ── Tokenization ──
"""
This stage is for both bangla-bert and mamba tokenization. If you use max_length: 512 then same tokenizer cache can be used by mamba and bangla-bert.
Otherwise, for different max_length you need to run this tokenizer separately for mamba and bangla-bert(set max_length in config.py)
"""
# stage = "Offline Tokenization"
# logging.info(f"Stage {stage} started")
# offline_tokenizer = OfflineTokenize()
# offline_tokenizer.initialize_tokenization()


# ── BanglaBERT Fine-Tuning (Steps 1–6: tokenizer load, dataset load, build, train) ──
stage = "BanglaBERT Fine-Tuning"
logging.info(f"Stage {stage} started")
bert_finetuner = BertFineTune()
bert_finetuner.initialize_bert_finetuning()   # also triggers BertEvaluate internally


# ── BanglaBERT Evaluation (Steps 7–10: standalone re-evaluation on saved best model) ──
# Useful if you want to re-run evaluation without re-training.
# stage = "BanglaBERT Evaluation"
# logging.info(f"Stage {stage} started")
# evaluator = BertEvaluate()
# evaluator.initialize_bert_evaluation(
#     test_loader      = ...,   # pass a pre-built DataLoader
#     tokenizer        = ...,   # pass a pre-loaded tokenizer
#     training_config  = {},    # optional metadata dict
#     training_history = [],    # optional per-epoch history
#     best_val_f1      = 0.0,   # optional scalar
# )

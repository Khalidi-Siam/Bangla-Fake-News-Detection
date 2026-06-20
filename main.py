from src.data_ingestion import DataIngestion
from src.offline_tokenize import OfflineTokenize
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
stage = "Offline Tokenization"
logging.info(f"Stage {stage} started")
offline_tokenizer = OfflineTokenize()
offline_tokenizer.initialize_tokenization()

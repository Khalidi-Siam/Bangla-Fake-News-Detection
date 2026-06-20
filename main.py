from src.data_ingestion import DataIngestion
from src.offline_tokenize import OfflineTokenize
from src.utils.logger import logging

# ── Data Ingestion ──
# stage = "Data Ingestion"
# logging.info(f"Stage {stage} started")
# data_ingestion = DataIngestion()
# data_ingestion.initialize_data_ingestion()

stage = "Offline Tokenization"
logging.info(f"Stage {stage} started")
offline_tokenizer = OfflineTokenize()
offline_tokenizer.initialize_tokenization()

import os
import sys
import io
import logging
from datetime import datetime
from config.config import settings

# VOLUME_PATH = "/root/datasets"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
# log_dir = os.path.join(VOLUME_PATH, "logs")
log_dir = settings.logging_dir
log_file_name = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, log_file_name)

# Wrap stdout in UTF-8 so Bangla (and any non-cp1252) text never crashes
# the StreamHandler on Windows. Use errors="replace" as a safe fallback.
utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    force=True,

    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(utf8_stdout),
    ]
)
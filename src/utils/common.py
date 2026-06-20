import yaml
from pathlib import Path
import os
import json
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys

from pathlib import Path
import logging

def create_directory(path: str | Path, verbose: bool = True):
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if verbose:
        logging.info(f"Created directory at: {path}")


def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def load_json(path: Path) -> dict:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        dict: data stored in json file
    """
    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded succesfully from: {path}")
    return content


def section(title: str):
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"  {title}")
    logging.info("=" * 60)
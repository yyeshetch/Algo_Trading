from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_level: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "engine.log"

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

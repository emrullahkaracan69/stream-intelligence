import logging
import logging.config
from pathlib import Path

from src.config import config


def setup_logging() -> None:
    """Configure application-wide logging.
    
    Logs are written to both console and a rotating file under LOG_DIR.
    Format includes timestamp, level, module name, and message.
    """
    
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "filename": str(log_dir / "stream_intelligence.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": config.log_level,
        },
    })
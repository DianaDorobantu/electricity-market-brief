
import logging
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "report.log"


def setup_logging(
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure application logging and return the report logger.
    """

    logger = logging.getLogger("report")
    logger.setLevel(log_level)
    logger.propagate = False

    # Avoid adding duplicate handlers if called more than once
    if not logger.handlers:
        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )

        logger.addHandler(file_handler)

    logger.info("Starting report for %s.", date.today())

    return logger

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the multi-agent research workflow."""

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("multi_agent_research")
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

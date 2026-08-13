"""Loguru setup shared by every stage CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route std-logging records (polars/pyarrow warnings) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(stage: str, run_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure loguru: stderr sink (+ optional per-run file sink), std-logging intercepted."""
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{extra[stage]}</cyan> | {message}"
    )
    logger.configure(extra={"stage": stage})
    logger.add(sys.stderr, level=level, format=fmt)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.add(run_dir / "log.txt", level="DEBUG", format=fmt, colorize=False)
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING, force=True)

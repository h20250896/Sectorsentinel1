from __future__ import annotations

import logging

from pythonjsonlogger.json import JsonFormatter


def configure_logging() -> None:
    formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)

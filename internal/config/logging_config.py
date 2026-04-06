import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Generator


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


class StructuredLogger:
    """A thin wrapper that emits JSON log records with arbitrary key-value fields."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        record = self._log.makeRecord(
            self._log.name, level, "(unknown)", 0, event, (), None
        )
        record.extra = kwargs  # type: ignore[attr-defined]
        self._log.handle(record)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, **kwargs)


@contextmanager
def timed(logger: StructuredLogger, event: str, **kwargs: Any) -> Generator[None, None, None]:
    """Context manager that logs *event* with a ``duration_ms`` field on exit."""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.error(event, duration_ms=elapsed_ms, error=str(exc), **kwargs)
        raise
    else:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.info(event, duration_ms=elapsed_ms, **kwargs)

import logging
import functools
import time
from typing import Any


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(funcName)s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a consistent format.
    Safe to call multiple times; subsequent calls will be no-ops if already configured.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT)

def attach_console_handler(level: int = logging.INFO) -> None:
    """Attach a console StreamHandler with our format regardless of existing handlers.
    Useful when frameworks (e.g., Streamlit) pre-configure logging and suppress INFO logs.
    """
    root = logging.getLogger()
    # Check if a StreamHandler with our format already exists
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)
            return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""
    return logging.getLogger(name)


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    try:
        s = repr(obj)
        return s if len(s) <= max_len else s[: max_len - 3] + "..."
    except Exception:
        return "<unrepr>"


def log_call(func):
    """Decorator that logs before/after a function call and on exceptions.
    Includes function name, module, and execution duration.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(
            f"START {func.__name__}",
        )
        logger.debug(
            f"ARGS {func.__name__}",
            extra={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)},
        )
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"END {func.__name__} ({duration:.3f}s)")
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.exception(f"ERROR {func.__name__} after {duration:.3f}s: {e}")
            raise
    return wrapper


def log_operation(logger: logging.Logger, name: str):
    """Context manager-style helper to log before/after an operation block.
    Usage:
        with log_operation(logger, "load_pdf"):
            ...
    """
    class _Op:
        def __enter__(self):
            self._start = time.perf_counter()
            logger.info(f"BEGIN {name}")

        def __exit__(self, exc_type, exc, tb):
            duration = time.perf_counter() - self._start
            if exc is None:
                logger.info(f"FINISH {name} ({duration:.3f}s)")
            else:
                logger.exception(f"FAIL {name} ({duration:.3f}s): {exc}")
            # Do not suppress exceptions
            return False

    return _Op()

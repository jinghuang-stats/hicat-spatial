"""Small, shared output helpers for pipeline stages."""

from __future__ import annotations

import contextlib
import functools
import json
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def ensure_output_dir(output_dir) -> Path:
    """Create a stage output directory and return it as a Path."""
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def save_json(value: Any, output_path) -> Path:
    """Save JSON-compatible metadata, safely stringifying custom objects."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2)
    return output_path


def save_stage_result(value: Any, output_path) -> Path:
    """Pickle a complete stage result for exact Python-level reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(value, handle)
    return output_path


def load_stage_result(input_path):
    """Load a result written by :func:`save_stage_result`."""
    with Path(input_path).expanduser().open("rb") as handle:
        return pickle.load(handle)


class _TeeStream:
    """Write text to the active console stream and a stage log file."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, text):
        self.console_stream.write(text)
        self.log_stream.write(text)
        return len(text)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def __getattr__(self, name):
        return getattr(self.console_stream, name)


@contextlib.contextmanager
def stage_log(output_dir, stage_name):
    """Tee stdout/stderr to a stage log and record elapsed wall-clock time.

    Files written inside ``output_dir`` are:

    - ``<stage_name>.log``: stdout, stderr, start/end timestamps, and errors.
    - ``timing.json``: machine-readable status and elapsed seconds.

    Notes
    -----
    ``redirect_stdout`` and ``redirect_stderr`` affect process-global streams,
    so stage runners should remain sequential rather than executing in threads.
    """
    output_dir = ensure_output_dir(output_dir)
    log_path = output_dir / f"{stage_name}.log"
    timing_path = output_dir / "timing.json"
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    status = "completed"
    error = None

    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        stdout_tee = _TeeStream(sys.stdout, log_handle)
        stderr_tee = _TeeStream(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(
            stderr_tee
        ):
            print(f"[{stage_name}] started at {started_at.isoformat()}")
            try:
                yield
            except BaseException as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{stage_name}] failed: {error}", file=sys.stderr)
                traceback.print_exc(file=log_handle)
                raise
            finally:
                ended_at = datetime.now(timezone.utc)
                elapsed_seconds = time.perf_counter() - started_clock
                print(
                    f"[{stage_name}] {status} at {ended_at.isoformat()} "
                    f"(elapsed {elapsed_seconds:.3f} seconds)"
                )
                save_json(
                    {
                        "stage": stage_name,
                        "status": status,
                        "started_at": started_at.isoformat(),
                        "ended_at": ended_at.isoformat(),
                        "elapsed_seconds": elapsed_seconds,
                        "error": error,
                        "log_path": str(log_path),
                    },
                    timing_path,
                )


def stage_output_from_config(
    default_output_dir,
    config_position,
    output_attribute="output_dir",
):
    """Build a decorator resolver for a positional/keyword stage config."""

    def resolver(*args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            config = args[config_position]
        output_dir = getattr(config, output_attribute)
        return output_dir or default_output_dir

    return resolver


def logged_stage(stage_name, output_dir_resolver):
    """Decorate a stage runner with console/file tee logging and timing."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            output_dir = output_dir_resolver(*args, **kwargs)
            with stage_log(output_dir=output_dir, stage_name=stage_name):
                return function(*args, **kwargs)

        return wrapper

    return decorator

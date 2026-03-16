"""
Time utilities for simulation.
"""

from datetime import datetime
from typing import Union


def str_to_timestamp(dt_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> float:
    """Convert datetime string to UNIX timestamp."""
    dt = datetime.strptime(dt_str, fmt)
    return dt.timestamp()


def timestamp_to_str(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert UNIX timestamp to formatted string."""
    dt = datetime.fromtimestamp(ts)
    return dt.strftime(fmt)


def format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"
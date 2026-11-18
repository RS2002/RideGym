"""Portable CSV order input with explicit units and time references.

Only the Python standard library is used. Coordinates retain the network's
units; numeric times are minutes. Timestamp input requires an explicit,
timezone-aware origin, rather than inferring a timezone or epoch unit.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Optional

from ride_gym.entities import Order

ORDER_COLUMNS = {
    "origin_x": "origin_x",
    "origin_y": "origin_y",
    "dest_x": "dest_x",
    "dest_y": "dest_y",
    "request_time": "request_time",
    "num_passengers": "num_passengers",
    "ready_time": "ready_time",
    "order_id": "order_id",
}
REQUIRED_ORDER_COLUMNS = ("origin_x", "origin_y", "dest_x", "dest_y", "request_time")


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise ValueError(f"{name} must be an integer")
    result = int(decimal)
    if positive and result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _timestamp(value, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp with a timezone") from exc
    else:
        raise ValueError(f"{name} must be an ISO timestamp with a timezone")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def _time(value, name: str, origin: Optional[datetime]) -> float:
    if origin is None:
        return _finite(value, name)
    timestamp = _timestamp(value, name)
    if timestamp.utcoffset() != origin.utcoffset():
        raise ValueError(f"{name} timezone offset must match time_origin")
    return _finite((timestamp - origin).total_seconds() / 60.0, name)


def _order_from_mapping(
    row: Mapping, columns: Mapping, default_id: int, origin: Optional[datetime] = None
) -> Order:
    def value(key):
        column = columns.get(key)
        return None if column is None else row.get(column)

    request = _time(value("request_time"), "request_time", origin)
    ready_value = value("ready_time")
    ready = None if _missing(ready_value) else _time(ready_value, "ready_time", origin)
    party_value = value("num_passengers")
    party = 1 if _missing(party_value) else _integer(party_value, "num_passengers", positive=True)
    id_column = columns.get("order_id")
    order_id = (
        default_id
        if id_column is None or id_column not in row
        else _integer(row[id_column], "order_id")
    )
    return Order(
        order_id=order_id,
        origin=(_finite(value("origin_x"), "origin_x"), _finite(value("origin_y"), "origin_y")),
        destination=(_finite(value("dest_x"), "dest_x"), _finite(value("dest_y"), "dest_y")),
        request_time=request,
        num_passengers=party,
        ready_time=ready,
    )


def read_order_csv(path, columns=None, time_origin=None) -> list[Order]:
    """Read fresh orders from a UTF-8 CSV, sorted by request time.

    Required columns are ``origin_x``, ``origin_y``, ``dest_x``, ``dest_y`` and
    ``request_time``. Optional ``num_passengers`` defaults to one; optional
    ``ready_time`` defaults to ``None``. ``order_id`` defaults to the original
    zero-based CSV row number when its column is absent. Explicit ids must be
    unique integers and are parsed without conversion through floating point.

    ``columns`` maps canonical names to custom CSV column names. Optional fields
    can be disabled with a ``None`` mapping. Extra CSV columns are ignored.

    With no ``time_origin``, times must be finite numeric minutes. With an aware
    datetime or ISO ``time_origin``, both time columns must contain ISO timestamps
    with the same UTC offset as the origin. No timezone or epoch unit is guessed.
    """
    supplied = {} if columns is None else columns
    if not isinstance(supplied, Mapping):
        raise TypeError("columns must be a mapping")
    unknown = set(supplied).difference(ORDER_COLUMNS)
    if unknown:
        raise ValueError("unknown canonical order columns: " + ", ".join(sorted(map(str, unknown))))
    mapping = {**ORDER_COLUMNS, **supplied}
    for key, name in mapping.items():
        if name is None and key not in REQUIRED_ORDER_COLUMNS:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError(f"column mapping for {key} must be a nonempty string")
    origin = None if time_origin is None else _timestamp(time_origin, "time_origin")
    orders, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            raise ValueError("order CSV contains duplicate column names")
        missing = [mapping[key] for key in REQUIRED_ORDER_COLUMNS if mapping[key] not in header]
        if missing:
            raise ValueError("order CSV is missing required columns: " + ", ".join(missing))
        for index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"CSV row {index + 2} has more cells than its header")
            try:
                order = _order_from_mapping(row, mapping, index, origin)
            except ValueError as exc:
                raise ValueError(f"CSV row {index + 2}: {exc}") from exc
            if order.order_id in seen:
                raise ValueError(f"CSV row {index + 2}: duplicate order_id")
            seen.add(order.order_id)
            orders.append(order)
    return sorted(orders, key=lambda order: order.request_time)

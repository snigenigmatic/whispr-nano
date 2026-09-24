"""Durable local reservation ledger for the bridge-open cloud budget.

Original reservations are immutable. Verified provider billing can replace the
amount counted toward the cap, retaining an explicit margin for late charges.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


HARD_CAP_USD = 15.0
DEFAULT_PATH = Path.home() / ".local/share/openresearch/whispr-bridge-open/budget-20260922.json"


class BudgetError(ValueError):
    """Raised when a budget operation cannot be accepted safely."""


_thread_locks: dict[str, threading.RLock] = {}
_thread_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _thread_locks_guard:
        return _thread_locks.setdefault(key, threading.RLock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BudgetError(f"{label} must be a nonempty string")


class BudgetLedger:
    """A fail-closed, atomically persisted reservation ledger."""

    def __init__(self, path: str | Path = DEFAULT_PATH, hard_cap_usd: float = HARD_CAP_USD):
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        if not math.isfinite(hard_cap_usd) or hard_cap_usd != HARD_CAP_USD:
            raise BudgetError(f"hard cap must be exactly {HARD_CAP_USD:g} USD")
        self.hard_cap_usd = float(hard_cap_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.path.exists():
                self._atomic_write(self._empty())
            else:
                self._check(self._read())

    def _empty(self) -> dict[str, Any]:
        return {"schema_version": 1, "hard_cap_usd": self.hard_cap_usd,
                "created_at": _now(), "reservations": {}}

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with _lock_for(self.path):
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text())
        except Exception as exc:
            raise BudgetError(f"cannot read budget ledger safely: {exc}") from exc
        self._check(value)
        return value

    def _check(self, value: Any) -> None:
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise BudgetError("invalid budget ledger schema")
        if value.get("hard_cap_usd") != self.hard_cap_usd:
            raise BudgetError("budget ledger hard cap mismatch")
        reservations = value.get("reservations")
        if not isinstance(reservations, dict):
            raise BudgetError("invalid reservations map")
        total = 0.0
        for job_id, reservation in reservations.items():
            if not isinstance(job_id, str) or not isinstance(reservation, dict):
                raise BudgetError("invalid reservation entry")
            amount = reservation.get("reserved_usd")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount) or amount <= 0:
                raise BudgetError(f"invalid reservation amount for {job_id}")
            settlement = reservation.get('settlement')
            if settlement is not None:
                held = settlement.get('held_usd')
                billed = settlement.get('provider_reported_usd')
                if (not isinstance(held, (int, float)) or isinstance(held, bool)
                    or not math.isfinite(held) or not isinstance(billed, (int, float))
                    or isinstance(billed, bool) or not math.isfinite(billed)
                    or billed < 0 or held < billed + 0.25 - 1e-9 or held > amount):
                    raise BudgetError('invalid provider settlement')
                for key in ['report_sha256', 'stopped_apps_sha256', 'app_id']:
                    _valid_hash(settlement.get(key), key)
            total += float(settlement['held_usd'] if settlement else amount)
        if total > self.hard_cap_usd + 1e-9:
            raise BudgetError("budget ledger exceeds hard cap")

    def _atomic_write(self, value: dict[str, Any]) -> None:
        data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            return self._read()

    def reserve(self, job_id: str, amount: float, config_hash: str) -> dict[str, Any]:
        _valid_hash(config_hash, "config_hash")
        if not isinstance(job_id, str) or not job_id.strip():
            raise BudgetError("job_id must be a nonempty string")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            raise BudgetError("reservation amount must be finite and positive")
        with self._locked():
            ledger = self._read()
            if job_id in ledger["reservations"]:
                raise BudgetError(f"duplicate job_id: {job_id}")
            total = sum(float(r.get('settlement', {}).get('held_usd', r['reserved_usd']))
                        for r in ledger['reservations'].values())
            if total + float(amount) > self.hard_cap_usd + 1e-9:
                raise BudgetError("reservation would exceed hard cap")
            reservation = {"job_id": job_id, "reserved_usd": float(amount),
                           "config_hash": config_hash, "status": "reserved",
                           "reserved_at": _now(), "estimated_resource_usd": None,
                           "receipt_hash": None}
            ledger["reservations"][job_id] = reservation
            self._atomic_write(ledger)
            return dict(reservation)

    def reconcile(self, job_id: str, app_id: str, report_path: str | Path,
                  stopped_apps_path: str | Path) -> dict[str, Any]:
        """Settle once using provider rows and a stopped-app receipt; retain $0.25."""
        from decimal import Decimal
        report_bytes = Path(report_path).read_bytes()
        stopped_bytes = Path(stopped_apps_path).read_bytes()
        report = json.loads(report_bytes)
        stopped = json.loads(stopped_bytes)
        app = next((a for a in stopped if a.get('app_id') == app_id), None)
        if not app or app.get('state') != 'stopped' or str(app.get('tasks')) != '0':
            raise BudgetError('provider app must be stopped with zero tasks')
        rows = [r for r in report if r.get('object_id') == app_id]
        if not rows or not any(r.get('resource') == 'H100' for r in rows):
            raise BudgetError('missing provider GPU billing rows')
        costs = [Decimal(r['cost']) for r in rows]
        if any(not c.is_finite() or c < 0 for c in costs):
            raise BudgetError('invalid provider costs')
        billed = sum(costs, Decimal(0))
        with self._locked():
            ledger = self._read()
            reservation = ledger['reservations'].get(job_id)
            if reservation is None or 'settlement' in reservation:
                raise BudgetError('unknown or already reconciled job')
            if any(r.get('settlement', {}).get('app_id') == app_id
                   for r in ledger['reservations'].values()):
                raise BudgetError('provider app already reconciled')
            held = float(billed + Decimal('0.25'))
            if held > reservation['reserved_usd']:
                raise BudgetError('provider charge plus margin exceeds original reservation')
            reservation['settlement'] = dict(app_id=app_id,
                provider_reported_usd=float(billed), held_usd=held,
                reconciled_at=_now(), report_path=str(Path(report_path).resolve()),
                report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                stopped_apps_sha256=hashlib.sha256(stopped_bytes).hexdigest())
            self._check(ledger)
            self._atomic_write(ledger)
            return dict(reservation)

    def record(self, job_id: str, status: str, estimated_resource_usd: float,
               receipt_hash: str) -> dict[str, Any]:
        if not isinstance(status, str) or not status.strip():
            raise BudgetError("status must be a nonempty string")
        if isinstance(estimated_resource_usd, bool) or not isinstance(estimated_resource_usd, (int, float)) or not math.isfinite(estimated_resource_usd) or estimated_resource_usd < 0:
            raise BudgetError("estimated resource cost must be finite and nonnegative")
        _valid_hash(receipt_hash, "receipt_hash")
        with self._locked():
            ledger = self._read()
            reservation = ledger["reservations"].get(job_id)
            if reservation is None:
                raise BudgetError(f"unknown job_id: {job_id}")
            if float(estimated_resource_usd) > float(reservation["reserved_usd"]) + 1e-9:
                raise BudgetError("estimated resource cost exceeds reservation")
            reservation.update(status=status, estimated_resource_usd=float(estimated_resource_usd),
                               receipt_hash=receipt_hash, recorded_at=_now())
            self._atomic_write(ledger)
            return dict(reservation)

"""Frozen study contract and fail-closed local cost ledger."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIRST_BATCH = (
    "smoke_source_pseudo_seed12",
    "source_real_seed12",
    "source_pseudo_seed12",
    "target_pseudo_seed12",
    "global_pseudo2real_seed12",
)


def canonical_json_hash(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class StudyContract:
    study_id: str
    code_branch: str
    seed: int
    student_model: str
    student_revision: str
    teacher_model: str
    teacher_revision: str
    updates: int
    batch_size: int
    learning_rate: float
    target_examples: int
    correction_scale: float
    sealed_test_speakers: int
    hard_cap_usd: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StudyContract":
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown contract fields: {unknown}")
        value = cls(**dict(raw))
        value.validate()
        return value

    @classmethod
    def load(cls, path: str | Path) -> "StudyContract":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.seed != 12:
            raise ValueError("the feasibility batch is frozen to seed 12")
        if self.code_branch != "codex/publication-study":
            raise ValueError("publication runs must use the publication branch")
        if self.updates != 600 or self.batch_size != 8:
            raise ValueError("first-batch training exposure is fixed at 600 x 8")
        if self.target_examples <= 0:
            raise ValueError("target_examples must be positive")
        for name in ("learning_rate", "correction_scale", "hard_cap_usd"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.hard_cap_usd != 12.0:
            raise ValueError("the user-authorized initial hard cap is US$12")
        if self.sealed_test_speakers != 198:
            raise ValueError("the frozen SPIRE test cohort must contain 198 speakers")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return canonical_json_hash(self.as_dict())


class BudgetLedger:
    """Reserve before dispatch against measured terminal and worst-case active cost.

    A reservation is deliberately conservative while a job is active or its
    cost is unknown. Once a terminal job records measured cost, the guard uses
    that amount. This prevents retries from silently exceeding the user's
    US$12 ceiling without treating unused reservation headroom as spending.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or raw.get("hard_cap_usd") != 12.0:
            raise ValueError("invalid publication budget ledger")
        raw.setdefault("jobs", [])
        return raw

    def reserved_total(self, raw: Mapping[str, Any] | None = None) -> float:
        data = dict(raw or self.load())
        return float(sum(float(job["reserved_usd"]) for job in data["jobs"]))

    def guarded_total(self, raw: Mapping[str, Any] | None = None) -> float:
        data = dict(raw or self.load())
        return float(sum(
            float(job.get("measured_usd", job["reserved_usd"]))
            for job in data["jobs"]
        ))

    def reserve(self, job_id: str, reserved_usd: float, config_sha256: str) -> dict[str, Any]:
        if not job_id or not config_sha256:
            raise ValueError("job_id and config_sha256 are required")
        if not math.isfinite(reserved_usd) or reserved_usd <= 0:
            raise ValueError("reserved_usd must be positive and finite")
        raw = self.load()
        if any(job["job_id"] == job_id for job in raw["jobs"]):
            raise ValueError(f"job id cannot be reused: {job_id}")
        total = self.guarded_total(raw) + reserved_usd
        if total > float(raw["hard_cap_usd"]) + 1e-9:
            raise RuntimeError(
                f"reservation would exceed US${raw['hard_cap_usd']:.2f}: US${total:.2f}"
            )
        raw["jobs"].append(
            {
                "job_id": job_id,
                "reserved_usd": reserved_usd,
                "config_sha256": config_sha256,
                "status": "reserved",
            }
        )
        atomic_json(self.path, raw)
        return raw["jobs"][-1]

    def complete(self, job_id: str, measured_usd: float, receipt_sha256: str) -> None:
        if not math.isfinite(measured_usd) or measured_usd < 0:
            raise ValueError("measured_usd must be non-negative and finite")
        raw = self.load()
        matches = [job for job in raw["jobs"] if job["job_id"] == job_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate job id: {job_id}")
        job = matches[0]
        if job["status"] != "reserved":
            raise ValueError(f"job is not reserved: {job_id}")
        if measured_usd > float(job["reserved_usd"]) + 1e-9:
            raise RuntimeError("measured cost exceeded reservation; freeze new dispatch")
        job.update(status="completed", measured_usd=measured_usd, receipt_sha256=receipt_sha256)
        atomic_json(self.path, raw)

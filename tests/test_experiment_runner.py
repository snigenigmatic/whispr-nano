"""The orx runner must accept only DistillConfig fields from run.json."""

from pathlib import Path

from onpolicy_distill.config import DistillConfig

ROOT = Path(__file__).resolve().parents[1]
RUN_JSON = ROOT / "experiments" / "run.json"


def test_committed_run_json_is_a_valid_distill_config():
    import json

    raw = json.loads(RUN_JSON.read_text())
    extra = set(raw) - set(DistillConfig.__dataclass_fields__)
    missing_required = []
    cfg = DistillConfig(**{k: v for k, v in raw.items() if k in DistillConfig.__dataclass_fields__})
    assert extra == set()
    assert not missing_required
    assert cfg.policy in {"on_policy", "off_policy", "scheduled_sampling"}
    assert 0.0 <= cfg.scheduled_sampling_prob <= 1.0

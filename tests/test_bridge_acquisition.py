import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bridge_open.acquire import timestamp


@pytest.mark.parametrize("value", ["00:60:00", "01:00:60", "1:2", "nan", -1, float("inf")])
def test_timestamp_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        timestamp(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("00:00:01.250", 1.25), ("01:02:03", 3723.0), (2.5, 2.5)],
)
def test_timestamp_parses_finite_nonnegative_times(value, expected):
    assert timestamp(value) == pytest.approx(expected)

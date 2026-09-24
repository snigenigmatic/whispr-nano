import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bridge_open.cloud_budget import BudgetError, BudgetLedger, HARD_CAP_USD


def test_cap_includes_active_and_completed_reservations(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("active", 10.0, "cfg-a")
    ledger.record("active", "completed", 1.0, "receipt-a")
    ledger.reserve("second", 5.0, "cfg-b")
    with pytest.raises(BudgetError):
        ledger.reserve("over", 0.01, "cfg-c")
    assert sum(r["reserved_usd"] for r in ledger.snapshot()["reservations"].values()) == HARD_CAP_USD


def test_duplicate_and_invalid_reservations_refuse(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("job", 1.0, "cfg")
    with pytest.raises(BudgetError):
        ledger.reserve("job", 1.0, "cfg")
    for amount in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(BudgetError):
            ledger.reserve(f"bad-{amount}", amount, "cfg")


def test_cost_breach_rejected_but_reservation_retained(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ledger.reserve("job", 2.0, "cfg")
    with pytest.raises(BudgetError, match="exceeds reservation"):
        ledger.record("job", "completed", 2.01, "receipt")
    saved = ledger.snapshot()["reservations"]["job"]
    assert saved["status"] == "reserved"
    assert saved["estimated_resource_usd"] is None


def test_concurrent_reservations_cannot_cross_cap(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    results = []

    def reserve(i):
        try:
            ledger.reserve(f"job-{i}", 1.0, f"cfg-{i}")
            results.append(True)
        except BudgetError:
            results.append(False)

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(results) == 15
    assert sum(r["reserved_usd"] for r in ledger.snapshot()["reservations"].values()) == HARD_CAP_USD


def test_provider_reconciliation_preserves_reservation_and_margin(tmp_path):
    import json
    ledger = BudgetLedger(tmp_path / 'budget.json')
    ledger.reserve('job', 15, 'config')
    report = tmp_path / 'provider.json'
    stopped = tmp_path / 'stopped.json'
    report.write_text(json.dumps([dict(object_id='ap-test', resource='H100', cost='1.20')]))
    stopped.write_text(json.dumps([dict(app_id='ap-test', state='stopped', tasks='0')]))
    settled = ledger.reconcile('job', 'ap-test', report, stopped)
    assert settled['reserved_usd'] == 15
    assert settled['settlement']['held_usd'] == 1.45
    ledger.reserve('next', 13.55, 'nextconfig')
    with pytest.raises(BudgetError):
        ledger.reserve('over', .01, 'x')
    with pytest.raises(BudgetError):
        ledger.reconcile('next', 'ap-test', report, stopped)


def test_reconciliation_refuses_running_and_missing_charges(tmp_path):
    import json
    ledger = BudgetLedger(tmp_path / 'budget.json')
    ledger.reserve('job', 1.5, 'config')
    report = tmp_path / 'provider.json'
    stopped = tmp_path / 'stopped.json'
    report.write_text('[]')
    stopped.write_text(json.dumps([dict(app_id='ap-test', state='running', tasks='1')]))
    with pytest.raises(BudgetError):
        ledger.reconcile('job', 'ap-test', report, stopped)
    stopped.write_text(json.dumps([dict(app_id='ap-test', state='stopped', tasks='0')]))
    with pytest.raises(BudgetError):
        ledger.reconcile('job', 'ap-test', report, stopped)
    assert 'settlement' not in ledger.snapshot()['reservations']['job']

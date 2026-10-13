"""Native App requests have no warm fleet and share the original budget cap."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/termeval'))
import launch_modal_app as m


def test_app_has_bounded_lifetime_and_no_warm_workers():
    o=m.function_options('L4',15)
    assert o['timeout']==900 and o['min_containers']==o['buffer_containers']==o['retries']==0
    assert o['max_containers']==1 and o['single_use_containers'] and o['scaledown_window']==2
    assert not o['serialized'] # Avoid host/image Python minor-version coupling.


def test_old_sandbox_reservations_still_count_towards_app_cap(tmp_path):
    ledger=tmp_path/'ledger.json'
    initial=dict(caps={'hf':20.,'modal':20.},reservations=[dict(provider='modal',reserved_usd=19.5)])
    ledger.write_text(json.dumps(initial))
    with patch.object(m,'LEDGER',ledger):
        with pytest.raises(SystemExit,match='budget refused'):m.reserve('exp','commit','L4',15,False)
    assert json.loads(ledger.read_text())==initial


def test_retry_refuses_any_possibly_dispatched_app(tmp_path):
    row=dict(id='hold',experiment='exp',commit='sha',provider='modal',runtime='App.function',flavor='L4',timeout_minutes=15,reserved_usd=.75)
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(reservations=[row])))
    (tmp_path/'modal-app-hold.json').write_text(json.dumps(dict(status='failed',app_id='ap-registered')))
    with patch.object(m,'LEDGER',ledger):
        with pytest.raises(SystemExit,match='possible worker dispatch'):
            m.retry_preflight('hold','exp','sha','L4',15)


def test_only_one_retry_can_claim_a_failed_preflight_hold(tmp_path):
    row=dict(id='hold',experiment='exp',commit='sha',provider='modal',runtime='App.function',flavor='L4',timeout_minutes=15,reserved_usd=.75)
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(reservations=[row])))
    (tmp_path/'modal-app-hold.json').write_text(json.dumps(dict(status='failed')))
    with patch.object(m,'LEDGER',ledger):
        claimed,_=m.retry_preflight('hold','exp','sha','L4',15)
        assert claimed==row
        with pytest.raises(SystemExit,match='possible worker dispatch'):
            m.retry_preflight('hold','exp','sha','L4',15)

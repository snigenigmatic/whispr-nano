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


def test_old_sandbox_reservations_still_count_towards_app_cap(tmp_path):
    ledger=tmp_path/'ledger.json'
    initial=dict(caps={'hf':20.,'modal':20.},reservations=[dict(provider='modal',reserved_usd=19.5)])
    ledger.write_text(json.dumps(initial))
    with patch.object(m,'LEDGER',ledger):
        with pytest.raises(SystemExit,match='budget refused'):m.reserve('exp','commit','L4',15,False)
    assert json.loads(ledger.read_text())==initial

"""The launcher must refuse overspend before it can call a cloud provider."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).parents[1]/'scripts/termeval/launch_budgeted_orx.py'
spec=importlib.util.spec_from_file_location('budgeted_orx',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CloudBudgetTest(unittest.TestCase):
    def call_with(self,provider,reservations):
        with tempfile.TemporaryDirectory() as td:
            ledger=Path(td)/'ledger.json'
            ledger.write_text(json.dumps(dict(schema=1,caps={'hf':20.,'modal':20.},reservations=reservations)))
            flavor='a100-large' if provider=='hf' else 'h100'
            argv=['launch','exp','--provider',provider,'--flavor',flavor,'--minutes','60','--dry-run']
            with patch.object(m,'LEDGER',ledger),patch.object(sys,'argv',argv),patch.object(m.subprocess,'run') as run:
                try:m.main()
                finally:run.assert_not_called()

    def test_over_cap_refused_without_provider_call(self):
        with self.assertRaisesRegex(SystemExit,'Budget refused'):
            self.call_with('hf',[dict(provider='hf',reserved_usd=19.25)])

    def test_provider_caps_are_separate(self):
        self.call_with('hf',[dict(provider='modal',reserved_usd=20.)])

    def test_invalid_time_never_estimated(self):
        for minutes in (0,-1,121,float('nan')):
            with self.assertRaises(ValueError):m.reservation('hf','a100-large',minutes)

if __name__=='__main__':unittest.main()

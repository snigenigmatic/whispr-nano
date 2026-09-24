"""Fixed experiment entry point with an isolated cloud worker."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
config_path = ROOT / "research/bridge_open/run_config.json"
if json.loads(config_path.read_text()).get("executor") == "evidence_snapshot":
    print("Evidence snapshot only. See research/bridge_open/SESSION_RESULTS_20260922.md for frozen ORX runs.")
    raise SystemExit(0)
if json.loads(config_path.read_text()).get("executor") == "modal":
    from research.bridge_open.cloud_launcher import main
    raise SystemExit(main())
RUNTIME = "/Users/prateekraj/whispr-nano/.venv-bridge/bin/python"
if Path(sys.executable).absolute() != Path(RUNTIME).absolute():
    os.execv(RUNTIME, [RUNTIME, __file__])
if json.loads(config_path.read_text()).get("executor") == "study_panels":
    from research.bridge_open.study_panels import main
    raise SystemExit(main())
if json.loads(config_path.read_text()).get("executor") == "analysis_expansion":
    from research.bridge_open.analyze_expansion import main
    raise SystemExit(main())
if json.loads(config_path.read_text()).get("executor") == "analysis":
    from research.bridge_open.analyze_runs import main
    raise SystemExit(main())
from research.bridge_open.run_whisper import run
raise SystemExit(run(config_path))

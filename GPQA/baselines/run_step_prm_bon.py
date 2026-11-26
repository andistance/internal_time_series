"""GPQA PRM Best-of-N at every step; reuse the existing GPQA evaluator."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.transformer_bon.step_entry import load_step_entry

_entry = load_step_entry(Path(__file__).with_name("run_prm_bon.py"), "GPQA.baselines._step_prm_entry")
build_parser, parse_args, run, _build_service = _entry.build_parser, _entry.parse_args, _entry.run, _entry._build_service
main = _entry.main

if __name__ == "__main__":
    main()

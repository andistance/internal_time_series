"""NaturalPlan trip step PRM Best-of-N using the original evaluator."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.transformer_bon.step_entry import load_step_entry

_entry = load_step_entry(Path(__file__).with_name("bon_runner_trip.py"), "NaturalPlan._step_trip")
run_bon = _entry.run_bon

def main(argv=None):
    run_bon("trip", "data/trip_planning.json", argv)

if __name__ == "__main__":
    main()

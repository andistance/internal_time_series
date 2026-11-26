"""GPQA multi-signal Transformer Best-of-N entry point.

The existing ``run_transformer_bon.py`` is untouched.  This wrapper reuses its
official GPQA loading, answer parsing, reporting, and nested Best-of-N logic,
and injects the reusable multi-signal service only at the service boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import run_transformer_bon as base_runner
except ImportError:
    import run_transformer_bon as base_runner

from scripts.transformer_bon_multi_signal.cli import add_multi_signal_arguments, build_service


def main() -> None:
    parser = base_runner.build_parser()
    add_multi_signal_arguments(parser, prefix="bon_")
    args = parser.parse_args()
    if args.bon_transformer_useless_token_percent not in [
        0.0,
        *[float(value) for value in args.bon_useless_token_percent],
    ]:
        parser.error("--bon_transformer_useless_token_percent must be one of --bon_useless_token_percent")
    service = build_service(args, prefix="bon_", n_name="N")
    try:
        metrics = base_runner.run(args, service=service)
    finally:
        service.close()
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Refusal fraction: {metrics['refusal_fraction']}")
    print(f"Results: {metrics['results_csv']}")


if __name__ == "__main__":
    main()

"""NaturalPlan multi-signal Transformer Best-of-N entry point.

The three benchmark runners in :mod:`NaturalPlan` intentionally remain
unchanged.  This thin launcher removes the extra multi-signal options before
delegating to the selected official runner and replaces only the service
constructor with :class:`MultiSignalBestOfNService`.  Consequently NaturalPlan
keeps its original parsing, correctness checks, nested-n reports, and output
format while sharing the reusable generation/forward/scoring implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.transformer_bon_multi_signal.cli import (
    DEFAULT_LEGACY_SCRIPT,
    parse_percentages,
)
from scripts.transformer_bon_multi_signal.config import MultiSignalBestOfNConfig
from scripts.transformer_bon_multi_signal.service import MultiSignalBestOfNService


_DEFAULT_DATA = {
    "calendar": str(ROOT / "deepscaler" / "data" / "test" / "calendar.json"),
    "meeting": str(ROOT / "deepscaler" / "data" / "test" / "meeting.json"),
    "trip": str(ROOT / "deepscaler" / "data" / "test" / "trip.json"),
}


def _add_multi_signal_arguments(parser: argparse.ArgumentParser) -> None:
    """Accept both NaturalPlan-style and ``bon_``-prefixed option names.

    NaturalPlan's original runner uses unprefixed Best-of-N arguments, while
    the GPQA/LiveCodeBench entry points use ``bon_`` prefixes.  Keeping both
    spellings here prevents the wrapper-only options from leaking into the
    original NaturalPlan parser as unrecognized arguments.
    """

    parser.add_argument(
        "--transformer_signal",
        "--bon_transformer_signal",
        dest="transformer_signal",
        choices=["logit", "prob", "entropy"],
        default="prob",
        help="Signal used as the value-transformer input.",
    )
    parser.add_argument(
        "--useless_token_percent",
        "--bon_useless_token_percent",
        dest="useless_token_percent",
        type=parse_percentages,
        default=[0.0],
        help="Comma-separated attention-pruning percentages; 0 is always included.",
    )
    parser.add_argument(
        "--transformer_useless_token_percent",
        "--bon_transformer_useless_token_percent",
        dest="transformer_useless_token_percent",
        type=float,
        default=0.0,
        help="Percentage used by the selected transformer signal.",
    )


def _service_factory(multi_args: argparse.Namespace):
    """Return a constructor compatible with the original NaturalPlan runner."""

    class Factory:
        def __new__(cls, base_config: Any) -> MultiSignalBestOfNService:
            # ``BestOfNConfig`` is a dataclass.  Filtering by the destination
            # dataclass fields also keeps this wrapper compatible with minor
            # additions to the legacy config in future revisions.
            allowed = {field.name for field in dataclasses.fields(MultiSignalBestOfNConfig)}
            values = {
                name: value
                for name, value in vars(base_config).items()
                if name in allowed
            }
            values.update(
                legacy_script_path=(values.get("legacy_script_path") or DEFAULT_LEGACY_SCRIPT),
                useless_token_percent=multi_args.useless_token_percent,
                transformer_signal=multi_args.transformer_signal,
                transformer_useless_token_percent=(
                    multi_args.transformer_useless_token_percent
                ),
            )
            return MultiSignalBestOfNService(MultiSignalBestOfNConfig(**values))

    return Factory


def _parse_launcher_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="NaturalPlan multi-signal Transformer Best-of-N options",
    )
    parser.add_argument("--benchmark", choices=sorted(_DEFAULT_DATA), default="calendar")
    parser.add_argument("--default_data_path", default=None)
    _add_multi_signal_arguments(parser)
    args, remaining = parser.parse_known_args(argv)
    percentages = [float(value) for value in args.useless_token_percent]
    if float(args.transformer_useless_token_percent) not in percentages:
        parser.error(
            "--transformer_useless_token_percent must be one of "
            "--useless_token_percent"
        )
    return args, remaining


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw_args or "-h" in raw_args:
        help_parser = argparse.ArgumentParser(
            description="NaturalPlan multi-signal Transformer Best-of-N options"
        )
        help_parser.add_argument("--benchmark", choices=sorted(_DEFAULT_DATA), default="calendar")
        help_parser.add_argument("--default_data_path")
        _add_multi_signal_arguments(help_parser)
        help_parser.print_help()
        return
    launcher_args, remaining = _parse_launcher_args(raw_args)
    if launcher_args.default_data_path:
        data_path = launcher_args.default_data_path
    else:
        data_path = _DEFAULT_DATA[launcher_args.benchmark]

    module = importlib.import_module(
        f"NaturalPlan.bon_runner_{launcher_args.benchmark}"
    )
    # Multi-signal trajectories are inputs to the confidence Transformer; the
    # original PRM backend has a different scoring contract.
    for score_option in ("--score_model", "--scoring_model"):
        if score_option not in remaining:
            continue
        try:
            score_pos = remaining.index(score_option)
            if score_pos + 1 < len(remaining) and remaining[score_pos + 1] != "transformer":
                raise SystemExit(
                    "NaturalPlan multi-signal mode requires --score_model transformer"
                )
        except ValueError:
            pass
    # The official runner chooses the class by its module-level symbol.  The
    # replacement is scoped to this call and is restored even after failures.
    original_service = module.BestOfNService
    module.BestOfNService = _service_factory(launcher_args)
    try:
        module.run_bon(launcher_args.benchmark, data_path, argv=remaining)
    finally:
        module.BestOfNService = original_service


if __name__ == "__main__":
    main()

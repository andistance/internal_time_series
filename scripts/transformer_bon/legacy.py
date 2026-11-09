"""Lazy access to the original research script.

The original file name starts with a digit, so it cannot be imported with a
normal Python import statement. Keeping this bridge small lets the reusable
service call the exact confidence/entropy/phase/model functions used by the
existing experiments instead of maintaining a numerically different copy.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional
from types import ModuleType


MULTI_SIGNAL_SCRIPT_NAME = "3_best_of_n_via_phases_scores_qwen_vllm_multi_signal.py"
MULTI_SIGNAL_IMPORT_NAME = "scripts.transformer_bon_multi_signal.legacy_runtime"


def resolve_legacy_script_path(explicit_path: Optional[str] = None) -> Path:
    """Resolve the canonical research script before any GPU worker starts."""

    configured = explicit_path or os.environ.get("TRANSFORMER_BON_LEGACY_SCRIPT")
    path = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1]
        / "3_best_of_n_via_phases_scores_qwen.py"
    )
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Transformer BoN requires the canonical legacy runtime script, but "
            f"it was not found at {path}. Copy "
            "3_best_of_n_via_phases_scores_qwen.py into the scripts directory "
            "or pass --bon_legacy_script_path/--legacy_script_path."
        )
    return path


@lru_cache(maxsize=None)
def _load_legacy_module_from_path(path_string: str) -> ModuleType:
    path = Path(path_string)
    # vLLM may create another multiprocessing layer inside a DP worker.  A
    # hash-based module name is sufficient for the parent process, but it is
    # not importable by a fresh ``spawn`` child during pickle unpickling.  The
    # multi-signal runtime therefore uses a real package module name whose
    # shim file re-executes the legacy script.  Existing single-signal scripts
    # retain the old isolated loading behavior.
    if path.name == MULTI_SIGNAL_SCRIPT_NAME:
        module_name = MULTI_SIGNAL_IMPORT_NAME
        shim_path = path.parent / "transformer_bon_multi_signal" / "legacy_runtime.py"
        spec = importlib.util.spec_from_file_location(module_name, shim_path)
    else:
        module_name = "_transformer_bon_legacy_" + str(abs(hash(path_string)))
        spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original Best-of-N module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            "Best-of-N runtime dependencies are unavailable. Install the same "
            "PyTorch, Transformers and vLLM environment used by the original script."
        ) from error
    return module


def load_legacy_module(explicit_path: Optional[str] = None) -> ModuleType:
    return _load_legacy_module_from_path(
        str(resolve_legacy_script_path(explicit_path))
    )

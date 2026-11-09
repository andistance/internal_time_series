from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from .aime import AIMEAdapter
from .field_mapped import FieldMappedAdapter
from .gpqa import GPQAAdapter


AdapterFactory = Callable[..., Any]
_REGISTRY: Dict[str, AdapterFactory] = {
    "aime": AIMEAdapter,
    "gpqa": GPQAAdapter,
    "field_mapped": FieldMappedAdapter,
}


def _create_icpc_adapter(**kwargs: Any) -> Any:
    from icpc_eval.bon_adapter import ICPCAdapter

    return ICPCAdapter(**kwargs)


_REGISTRY["icpc_eval"] = _create_icpc_adapter


def register_adapter(name: str, factory: AdapterFactory) -> None:
    if not name:
        raise ValueError("Adapter name must not be empty")
    _REGISTRY[name] = factory


def create_adapter(name_or_import_path: str, **kwargs: Any) -> Any:
    if name_or_import_path in _REGISTRY:
        return _REGISTRY[name_or_import_path](**kwargs)
    if ":" not in name_or_import_path:
        raise KeyError(
            f"Unknown dataset adapter {name_or_import_path!r}; registered: "
            f"{sorted(_REGISTRY)}"
        )
    module_name, attribute_name = name_or_import_path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    return factory(**kwargs)

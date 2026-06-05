"""Backend registry — maps --backend flag values to backend classes.

Backend naming convention:  <tm_variant>_<architecture>

  vanilla_tiled      — vanilla TM, tiled FSM + tile ROM
  vanilla_hardwired  — vanilla TM, HCB streaming + adder tree

Future backends follow the same convention:
  coalesced_tiled, weighted_hardwired, convolutional_tiled, …

Backends are lazy-loaded on first use so importing the registry never
triggers RTL-generator or emulator imports.
"""

from __future__ import annotations

import importlib

# name -> (module_path, class_name)  — loaded lazily
_BACKENDS: dict[str, tuple[str, str]] = {
    "vanilla_tiled":     ("matador.backends.tiled.rtl",     "TiledBackend"),
    "vanilla_hardwired": ("matador.backends.hardwired.rtl", "HardwiredBackend"),
}

# Short one-line descriptions shown in list-backends and the splash screen
_DESCRIPTIONS: dict[str, str] = {
    "vanilla_tiled":     "Vanilla TM — sequential FSM + tile ROM. Knobs: feat_slice, clause_slice.",
    "vanilla_hardwired": "Vanilla TM — HCB streaming + adder tree.  Knobs: pipeline_stages.",
}


def get(name: str):
    """Return the RTLBackend *class* for the given name.

    Raises:
        ValueError: if the name is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown backend: {name!r}. "
            f"Available backends: {available}"
        )
    module_path, class_name = _BACKENDS[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_BACKENDS.keys())


def describe(name: str) -> str:
    """Return the one-line description for a registered backend."""
    return _DESCRIPTIONS.get(name, "")


def config_class_for(name: str) -> type:
    """Return the Pydantic config class for the named backend."""
    return get(name)().config_class

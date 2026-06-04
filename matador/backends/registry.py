"""Backend registry — maps --backend flag values to backend classes.

Backends are lazy-loaded on first use so importing the registry never
triggers RTL-generator or emulator imports.

Built-in backends:
  tiled      — tiled feature-clause matrix (sequential FSM, tile ROM)
  hardwired  — combinational AND-gate unrolling (adder tree, no ROM)  [Phase 2]
"""

from __future__ import annotations

import importlib

# name -> (module_path, class_name)  — loaded lazily
_BACKENDS: dict[str, tuple[str, str]] = {
    "tiled":     ("matador.backends.tiled.rtl",     "TiledBackend"),
    "hardwired": ("matador.backends.hardwired.rtl", "HardwiredBackend"),
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


def config_class_for(name: str) -> type:
    """Return the Pydantic config class for the named backend."""
    return get(name)().config_class

"""Tiled backend emulator — re-exports TMAcceleratorEmulator under the
backend-namespaced name TiledEmulator.

The concrete implementation lives in matador.emulator.accelerator and will
migrate here in a future cleanup.
"""

from matador.emulator.accelerator import TMAcceleratorEmulator as TiledEmulator

__all__ = ["TiledEmulator"]

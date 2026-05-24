"""Hardware-mirroring software emulator for the TM inference accelerator."""

from matador.emulator.accelerator import TMAcceleratorEmulator
from matador.emulator.trace import InferenceTrace

__all__ = ["TMAcceleratorEmulator", "InferenceTrace"]

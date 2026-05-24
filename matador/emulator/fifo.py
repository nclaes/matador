"""FIFOEmulator — circular-buffer AXI-Stream FIFO mirroring axis_fifo.v."""

from __future__ import annotations

from matador.emulator.trace import FIFOPopEvent, FIFOPushEvent


class FIFOEmulator:
    """Circular-buffer FIFO with AXI-Stream TVALID/TREADY/TLAST protocol.

    Mirrors axis_fifo.v: power-of-2 depth, wptr/rptr, full/empty flags.
    """

    def __init__(self, depth: int, cycle_ref: list):
        """
        Args:
            depth:      FIFO depth (must be power of 2).
            cycle_ref:  Single-element list holding current cycle counter
                        (shared mutable reference with the FSM orchestrator).
        """
        if depth < 1 or (depth & (depth - 1)) != 0:
            raise ValueError(f"FIFO depth must be a power of 2, got {depth}")
        self._depth = depth
        self._mask = depth - 1
        self._buf_data = [0] * depth
        self._buf_last = [False] * depth
        self._wptr = 0
        self._rptr = 0
        self._count = 0
        self._cycle_ref = cycle_ref
        self.events: list = []

    @property
    def occupancy(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self._depth

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    def push(self, data: int, tlast: bool) -> FIFOPushEvent:
        if self.is_full:
            raise RuntimeError("FIFO overflow — push into full FIFO")
        self._buf_data[self._wptr & self._mask] = data
        self._buf_last[self._wptr & self._mask] = tlast
        self._wptr = (self._wptr + 1) & self._mask
        self._count += 1
        ev = FIFOPushEvent(
            cycle=self._cycle_ref[0],
            data=data, tlast=tlast,
            wptr=self._wptr, occupancy=self._count,
        )
        self.events.append(ev)
        return ev

    def pop(self) -> tuple:
        """Return (data, tlast) and emit FIFOPopEvent."""
        if self.is_empty:
            raise RuntimeError("FIFO underflow — pop from empty FIFO")
        data = self._buf_data[self._rptr & self._mask]
        tlast = self._buf_last[self._rptr & self._mask]
        self._rptr = (self._rptr + 1) & self._mask
        self._count -= 1
        ev = FIFOPopEvent(
            cycle=self._cycle_ref[0],
            data=data, tlast=tlast,
            rptr=self._rptr, occupancy=self._count,
        )
        self.events.append(ev)
        return data, tlast, ev

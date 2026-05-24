"""ArgmaxEmulator — pure function mirroring argmax.v tournament tree."""

from __future__ import annotations

from matador.emulator.trace import ArgmaxEvent


class ArgmaxEmulator:
    """Combinatorial argmax over a list of signed scores.

    Ties broken by lower index (matches RTL tournament-tree tie-breaking).
    """

    def argmax(self, scores: list, cycle: int = 0) -> tuple:
        """Return (predicted_class, ArgmaxEvent).

        Args:
            scores: list of signed integer scores, one per class.
            cycle:  current cycle counter for the event.
        """
        best_idx = 0
        best_val = scores[0]
        for i in range(1, len(scores)):
            if scores[i] > best_val:
                best_val = scores[i]
                best_idx = i
        ev = ArgmaxEvent(cycle=cycle, scores=scores, predicted_class=best_idx)
        return best_idx, ev

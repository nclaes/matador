"""ScoreAccEmulator — stateful score accumulator mirroring score_acc.v."""

from __future__ import annotations

from matador.emulator.trace import ScoreStateEvent, ScoreVoteEvent


class ScoreAccEmulator:
    """Signed score accumulator with symmetric saturation at ±threshold.

    Mirrors score_acc.v:
      - Positive clause vote: score[cls] = min(score[cls] + 1, +T)
      - Negative clause vote: score[cls] = max(score[cls] - 1, -T)
      - Only updates when clause_active == True
    """

    def __init__(self, n_classes: int, threshold: int, cycle_ref: list):
        self._n_classes = n_classes
        self._threshold = threshold
        self._scores = [0] * n_classes
        self._cycle_ref = cycle_ref
        self._vote_cnt = 0

    def clear(self) -> None:
        self._scores = [0] * self._n_classes
        self._vote_cnt = 0

    def get_scores(self) -> list:
        return list(self._scores)

    def update(self, class_idx: int, is_positive: bool, clause_active: bool) -> ScoreVoteEvent:
        """Apply one clause vote and return the event."""
        before = self._scores[class_idx]
        if clause_active:
            if is_positive:
                after = min(before + 1, self._threshold)
            else:
                after = max(before - 1, -self._threshold)
            saturated = (after == before) and clause_active
        else:
            after = before
            saturated = False
        self._scores[class_idx] = after

        polarity = "positive" if is_positive else "negative"
        ev = ScoreVoteEvent(
            cycle=self._cycle_ref[0],
            score_cnt=self._vote_cnt,
            clause_global=self._vote_cnt,
            class_idx=class_idx,
            polarity=polarity,
            clause_active=clause_active,
            score_before=before,
            score_after=after,
            saturated=saturated,
        )
        self._vote_cnt += 1
        return ev

    def state_snapshot(self) -> ScoreStateEvent:
        return ScoreStateEvent(cycle=self._cycle_ref[0], scores=self.get_scores())

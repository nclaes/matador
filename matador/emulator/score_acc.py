"""ScoreAccEmulator — stateful score accumulator mirroring score_acc.v."""

from __future__ import annotations

from matador.emulator.trace import ScoreStateEvent, ScoreVoteEvent


class ScoreAccEmulator:
    """Signed score accumulator, unclamped — the raw vote sum.

    Mirrors score_acc.v:
      - Positive clause vote: score[cls] = score[cls] + 1
      - Negative clause vote: score[cls] = score[cls] - 1
      - Only updates when clause_active == True

    No saturation: the register in score_acc.v is sized (score_width, see
    TMAccelerator) to the true worst-case vote magnitude for the model
    being generated, so it can never overflow -- there is nothing to clamp
    against. This matches the software reference engine
    (matador.inference.reference.predict) exactly, which also never clips
    the vote sum; clause-sum clipping to +/-T only exists in the Tsetlin
    Machine literature as a TRAINING-time feedback-probability construct,
    not part of inference/classification.
    """

    def __init__(self, n_classes: int, cycle_ref: list):
        self._n_classes = n_classes
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
        after = (before + 1 if is_positive else before - 1) if clause_active else before
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
        )
        self._vote_cnt += 1
        return ev

    def state_snapshot(self) -> ScoreStateEvent:
        return ScoreStateEvent(cycle=self._cycle_ref[0], scores=self.get_scores())

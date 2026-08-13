import numpy as np

from coal_tm.emulator import CoalescedEmulator


def _emu(model):
    return CoalescedEmulator(model["tas"], model["weights"], model["classes"], model["clauses"], model["features"])


def _x(f0: int, f1: int) -> np.ndarray:
    """A 16-feature vector with only f0/f1 set; the rest are always 0 and
    always excluded, so they can't affect any clause in the tiny model."""
    x = np.zeros(16, dtype=np.uint8)
    x[0], x[1] = f0, f1
    return x


def test_clause_fires_when_its_literals_are_satisfied(tiny_model):
    emu = _emu(tiny_model)
    result = emu.predict(_x(1, 0))  # x0=1, x1=0
    assert list(result.clause_outputs) == [1, 0]
    assert list(result.class_sums) == [5, -3]
    assert result.predicted_class == 0


def test_clause_does_not_fire_when_a_literal_is_unsatisfied(tiny_model):
    emu = _emu(tiny_model)
    result = emu.predict(_x(1, 1))  # x1=1 violates clause 0's ~x1
    assert list(result.clause_outputs) == [0, 0]


def test_all_exclude_clause_is_forced_false_not_vacuously_true(tiny_model):
    """Clause 1 has zero included literals. If it were (incorrectly) treated
    as vacuously true, its weight (100/7 -- deliberately large) would
    dominate every class sum. TMU's own reference forces empty clauses to 0
    (ClauseBank.c: "Make empty clauses false")."""
    emu = _emu(tiny_model)
    result = emu.predict(_x(0, 0))
    assert result.clause_outputs[1] == 0
    assert list(result.class_sums) == [0, 0]


def test_predict_batch_matches_predict(tiny_model):
    emu = _emu(tiny_model)
    X = np.array([_x(1, 0), _x(1, 1), _x(0, 0)], dtype=np.uint8)
    predictions, sums = emu.predict_batch(X)
    for i, x in enumerate(X):
        single = emu.predict(x)
        assert predictions[i] == single.predicted_class
        assert list(sums[i]) == list(single.class_sums)

import pytest
from pydantic import ValidationError

from matador.config.schema import TMType, TrainingConfig

_VALID = dict(
    tm_type="vanilla",
    clauses=400,
    classes=10,
    features=512,
    s=5.0,
    T=200,
    epochs=50,
    max_included_literals=32,
    seed=42,
    train_data="/tmp/train.txt",
    test_data="/tmp/test.txt",
    output_dir="/tmp/output",
)


def _cfg(**overrides) -> TrainingConfig:
    return TrainingConfig.model_validate({**_VALID, **overrides})


def test_valid_config_parses():
    cfg = _cfg()
    assert cfg.tm_type is TMType.vanilla
    assert cfg.clauses == 400


def test_coalesced_type_accepted():
    cfg = _cfg(tm_type="coalesced")
    assert cfg.tm_type is TMType.coalesced


def test_invalid_tm_type_rejected():
    with pytest.raises(ValidationError, match="tm_type"):
        _cfg(tm_type="unknown")


def test_odd_clauses_rejected():
    with pytest.raises(ValidationError, match="even"):
        _cfg(clauses=401)


def test_zero_clauses_rejected():
    with pytest.raises(ValidationError):
        _cfg(clauses=0)


def test_s_must_be_greater_than_one():
    with pytest.raises(ValidationError):
        _cfg(s=1.0)


def test_max_literals_exceeds_features_rejected():
    with pytest.raises(ValidationError, match="max_included_literals"):
        _cfg(features=4, max_included_literals=9)


def test_default_seed():
    cfg = TrainingConfig.model_validate({k: v for k, v in _VALID.items() if k != "seed"})
    assert cfg.seed == 42

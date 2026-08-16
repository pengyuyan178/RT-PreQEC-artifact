import numpy as np

from rt_preqec.predecode.validator import validate_local_correction


def test_validator_pass() -> None:
    patch = np.zeros((3, 5, 5), dtype=np.int8)
    patch[-1, 2, 2] = 1
    correction = np.zeros(25, dtype=np.int8)
    correction[12] = 1
    assert validate_local_correction(patch, correction)


def test_validator_fail() -> None:
    patch = np.zeros((3, 5, 5), dtype=np.int8)
    patch[-1, 2, 2] = 1
    correction = np.zeros(25, dtype=np.int8)
    correction[0] = 1
    assert not validate_local_correction(patch, correction)

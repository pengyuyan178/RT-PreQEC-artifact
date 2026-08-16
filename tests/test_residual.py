import numpy as np

from rt_preqec.predecode.residual import apply_local_corrections_to_syndrome, compute_residual_density


def test_residual_density_reduction() -> None:
    syndrome = np.zeros((3, 5, 5), dtype=np.int8)
    syndrome[-1, 2, 2] = 1
    correction = np.zeros(25, dtype=np.int8)
    correction[12] = 1
    residual = apply_local_corrections_to_syndrome(syndrome, [correction], [(2, 2, 2)])
    assert compute_residual_density(syndrome, residual) < 1.0


def test_residual_shape_match() -> None:
    syndrome = np.zeros((3, 5, 5), dtype=np.int8)
    correction = np.zeros(25, dtype=np.int8)
    residual = apply_local_corrections_to_syndrome(syndrome, [correction], [(2, 2, 2)])
    assert residual.shape == syndrome.shape

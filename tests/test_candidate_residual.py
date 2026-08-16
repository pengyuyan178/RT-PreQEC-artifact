import numpy as np

from rt_preqec.data.schemas import LocalErrorCandidate
from rt_preqec.predecode.residual import apply_candidate_to_flat_syndrome


def test_apply_candidate_to_flat_syndrome() -> None:
    syndrome = np.asarray([0, 1, 1, 0], dtype=np.int8)
    candidate = LocalErrorCandidate(0, np.asarray([1, 2], dtype=np.int32), np.asarray([], dtype=np.int32), 0.2, 1.0, {}, {})
    residual = apply_candidate_to_flat_syndrome(syndrome, candidate)
    assert residual.tolist() == [0, 0, 0, 0]

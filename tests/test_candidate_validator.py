import numpy as np

from rt_preqec.data.schemas import DetectorPatch, LocalErrorCandidate
from rt_preqec.predecode.validator import validate_candidate_against_patch


def _patch() -> DetectorPatch:
    return DetectorPatch(
        patch_id=0,
        shot_id=0,
        center_detector_id=1,
        detector_ids=np.asarray([0, 1, 2], dtype=np.int32),
        detector_coords=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
        syndrome_bits=np.asarray([0, 1, 1], dtype=np.int8),
        active_detector_ids=np.asarray([1, 2], dtype=np.int32),
        metadata={},
    )


def test_candidate_exact_match_passes() -> None:
    candidate = LocalErrorCandidate(0, np.asarray([1, 2], dtype=np.int32), np.asarray([], dtype=np.int32), 0.2, 1.0, {}, {})
    result = validate_candidate_against_patch(_patch(), candidate, require_exact_match=True)
    assert result.passed


def test_candidate_subset_passes_non_exact() -> None:
    candidate = LocalErrorCandidate(0, np.asarray([1], dtype=np.int32), np.asarray([], dtype=np.int32), 0.2, 1.0, {}, {})
    result = validate_candidate_against_patch(_patch(), candidate, require_exact_match=False)
    assert result.passed


def test_candidate_with_observable_fails() -> None:
    candidate = LocalErrorCandidate(0, np.asarray([1, 2], dtype=np.int32), np.asarray([0], dtype=np.int32), 0.2, 1.0, {}, {})
    result = validate_candidate_against_patch(_patch(), candidate, allow_observable_flip=False)
    assert not result.passed

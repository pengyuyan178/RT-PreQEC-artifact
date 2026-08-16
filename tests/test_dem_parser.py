import importlib.util

import numpy as np

from rt_preqec.data.dem_parser import parse_dem_error_candidates
from rt_preqec.data.layout import build_detector_layout_from_dem


def test_dem_parser_handles_missing_input() -> None:
    assert parse_dem_error_candidates(None) == []


def test_dem_parser_small_circuit_if_available() -> None:
    if importlib.util.find_spec("stim") is None:
        assert parse_dem_error_candidates(None) == []
        return
    import stim

    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.001)
    dem = circuit.detector_error_model(decompose_errors=True)
    layout = build_detector_layout_from_dem(dem)
    candidates = parse_dem_error_candidates(dem, layout=layout)
    assert isinstance(candidates, list)
    if candidates:
        assert isinstance(candidates[0].detector_ids, np.ndarray)

import numpy as np

from rt_preqec.data.layout import DetectorCoord, DetectorLayout, build_spatial_index
from rt_preqec.data.patch_extractor import extract_detector_patches_from_flat_syndrome


def test_extract_detector_patches_from_flat_syndrome() -> None:
    layout = build_spatial_index(
        DetectorLayout(
            coords=[
                DetectorCoord(detector_id=0, raw_coord=[0.0, 0.0, 0.0], inferred_time=0.0, inferred_x=0.0, inferred_y=0.0),
                DetectorCoord(detector_id=1, raw_coord=[0.0, 1.0, 0.0], inferred_time=0.0, inferred_x=1.0, inferred_y=0.0),
                DetectorCoord(detector_id=2, raw_coord=[0.0, 2.0, 0.0], inferred_time=0.0, inferred_x=2.0, inferred_y=0.0),
                DetectorCoord(detector_id=3, raw_coord=[1.0, 0.0, 0.0], inferred_time=1.0, inferred_x=0.0, inferred_y=0.0),
                DetectorCoord(detector_id=4, raw_coord=[1.0, 1.0, 0.0], inferred_time=1.0, inferred_x=1.0, inferred_y=0.0),
            ]
        )
    )
    syndrome = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    patches = extract_detector_patches_from_flat_syndrome(syndrome, layout, patch_radius=1.5, time_radius=1.0)
    assert len(patches) > 0
    patch = patches[0]
    assert np.all(patch.syndrome_bits == syndrome[patch.detector_ids])
    assert set(patch.active_detector_ids.tolist()).issubset(set(np.flatnonzero(syndrome).tolist()))

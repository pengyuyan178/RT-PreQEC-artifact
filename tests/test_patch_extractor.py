import numpy as np

from rt_preqec.data.patch_extractor import extract_local_patches


def test_extract_patches_non_empty() -> None:
    syndrome = np.random.randint(0, 2, size=(3, 5, 5), dtype=np.int8)
    patches = extract_local_patches(syndrome, patch_size=5, temporal_window=3)
    assert len(patches) > 0


def test_patch_shape_correct() -> None:
    syndrome = np.random.randint(0, 2, size=(3, 5, 5), dtype=np.int8)
    patches = extract_local_patches(syndrome, patch_size=5, temporal_window=3)
    assert patches[0]["patch"].shape == (3, 5, 5)

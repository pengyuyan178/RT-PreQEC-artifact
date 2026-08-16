import numpy as np

from rt_preqec.data.risk_features import extract_syndrome_features


def test_extract_syndrome_features_fixed_length_and_no_nan() -> None:
    syndrome = np.random.randint(0, 2, size=(17,), dtype=np.int8)
    features, names = extract_syndrome_features(syndrome)
    assert features.ndim == 1
    assert len(features) == len(names)
    assert len(features) > 0
    assert not np.isnan(features).any()

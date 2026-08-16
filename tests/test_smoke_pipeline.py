import numpy as np

from rt_preqec.config import load_config
from rt_preqec.data.dem_parser import index_candidates_by_detector
from rt_preqec.data.layout import DetectorCoord, DetectorLayout, build_spatial_index
from rt_preqec.data.schemas import LocalErrorCandidate
from rt_preqec.decoders.lookup_decoder import LookupDecoder
from rt_preqec.decoders.pymatching_decoder import PyMatchingDecoder
from rt_preqec.predecode.selective_predecoder import SelectivePredecoder
from rt_preqec.runtime.pipeline import RTPreQECPipeline
from rt_preqec.runtime.stream_simulator import SyndromeStreamSimulator
from rt_preqec.scheduler.lag_scheduler import LagBoundedScheduler


def test_smoke_pipeline_runs() -> None:
    config = load_config("configs/eval_realtime.yaml")
    predecoder = SelectivePredecoder(
        model=None,
        confidence_threshold=config.predecoder.confidence_threshold,
        risk_threshold=config.predecoder.risk_threshold,
        correction_threshold=config.predecoder.correction_threshold,
        enable_validation=config.predecoder.enable_validation,
        enable_abstention=config.predecoder.enable_abstention,
        device=config.device,
    )
    pipeline = RTPreQECPipeline(
        config,
        predecoder,
        LagBoundedScheduler(config),
        {"lookup": LookupDecoder(), "pymatching": PyMatchingDecoder()},
    )
    syndromes = [np.random.randint(0, 2, size=(3, 5, 5), dtype=np.int8) for _ in range(8)]
    simulator = SyndromeStreamSimulator(config.runtime.round_period_us, config.runtime.decode_deadline_us)
    metrics = pipeline.run_stream(simulator.from_syndromes(syndromes))
    assert "deadline_miss_ratio" in metrics
    assert "latency" in metrics
    assert "backlog" in metrics
    assert "accept_rate" in metrics


def test_layout_aware_smoke_pipeline_runs() -> None:
    config = load_config("configs/eval_realtime.yaml")
    layout = build_spatial_index(
        DetectorLayout(
            coords=[
                DetectorCoord(detector_id=0, raw_coord=[0.0, 0.0, 0.0], inferred_time=0.0, inferred_x=0.0, inferred_y=0.0),
                DetectorCoord(detector_id=1, raw_coord=[0.0, 1.0, 0.0], inferred_time=0.0, inferred_x=1.0, inferred_y=0.0),
                DetectorCoord(detector_id=2, raw_coord=[0.0, 2.0, 0.0], inferred_time=0.0, inferred_x=2.0, inferred_y=0.0),
                DetectorCoord(detector_id=3, raw_coord=[1.0, 0.0, 0.0], inferred_time=1.0, inferred_x=0.0, inferred_y=0.0),
            ]
        )
    )
    candidates = [
        LocalErrorCandidate(
            candidate_id=0,
            detector_ids=np.asarray([1, 2], dtype=np.int32),
            observable_ids=np.asarray([], dtype=np.int32),
            probability=0.2,
            weight=1.0,
            coord_span={"spatial_diameter": 1.0, "time_diameter": 0.0},
            metadata={},
        )
    ]
    predecoder = SelectivePredecoder(
        model=None,
        confidence_threshold=0.1,
        risk_threshold=0.9,
        correction_threshold=config.predecoder.correction_threshold,
        enable_validation=True,
        enable_abstention=True,
        device=config.device,
        layout=layout,
        candidates_by_detector=index_candidates_by_detector(candidates),
        mode="candidate",
    )
    pipeline = RTPreQECPipeline(
        config,
        predecoder,
        LagBoundedScheduler(config),
        {"lookup": LookupDecoder(), "pymatching": PyMatchingDecoder()},
    )
    syndromes = [np.asarray([0, 1, 1, 0], dtype=np.int8) for _ in range(4)]
    simulator = SyndromeStreamSimulator(config.runtime.round_period_us, config.runtime.decode_deadline_us)
    metrics = pipeline.run_stream(
        simulator.from_flat_syndromes(
            syndromes,
            layout=layout,
            extra_metadata={"patch_radius": 2.0, "time_radius": 1.0},
        )
    )
    assert "deadline_miss_ratio" in metrics
    assert "accept_rate" in metrics

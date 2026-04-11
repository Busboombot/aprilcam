"""Core detection engine, data models, and playfield geometry."""

from .motion import VelocityEstimator
from .detector import TagDetector, DetectorConfig, Detection
from .tracker import OpticalFlowTracker
from .tag import Tag
from .pipeline import DetectionPipeline
from .playfield import Playfield, PlayfieldBoundary
from .detection import TagRecord, FrameRecord, RingBuffer

__all__ = [
    "VelocityEstimator",
    "TagDetector",
    "DetectorConfig",
    "Detection",
    "OpticalFlowTracker",
    "Tag",
    "DetectionPipeline",
    "Playfield",
    "PlayfieldBoundary",
    "TagRecord",
    "FrameRecord",
    "RingBuffer",
]

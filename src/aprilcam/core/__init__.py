"""Core detection engine, data models, and playfield geometry."""

from .motion import VelocityEstimator
from .detector import TagDetector, DetectorConfig, Detection
from .tracker import OpticalFlowTracker

__all__ = [
    "VelocityEstimator",
    "TagDetector",
    "DetectorConfig",
    "Detection",
    "OpticalFlowTracker",
]

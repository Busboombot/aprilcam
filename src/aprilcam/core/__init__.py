"""Core detection engine, data models, and playfield geometry."""

from .motion import VelocityEstimator
from .detector import TagDetector, DetectorConfig, Detection

__all__ = [
    "VelocityEstimator",
    "TagDetector",
    "DetectorConfig",
    "Detection",
]

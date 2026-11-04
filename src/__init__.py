"""MQ-4 Enclosure Detection Package.

YOLOv8n-based object detection application for Raspberry Pi edge deployment.
"""

from src.camera import run_camera_stream
from src.config import DEFAULT_CLASS_COLORS, DEFAULT_CLASS_NAMES, DetectorConfig, get_project_root
from src.detector import DetectionResult, MQ4Detector
from src.image_inference import run_image_inference
from src.utils import benchmark_model, create_synthetic_test_image, get_system_telemetry

__all__ = [
    "MQ4Detector",
    "DetectionResult",
    "DetectorConfig",
    "get_project_root",
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_CLASS_COLORS",
    "run_image_inference",
    "run_camera_stream",
    "benchmark_model",
    "create_synthetic_test_image",
    "get_system_telemetry",
]

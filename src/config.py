"""Centralized configuration module for MQ-4 Enclosure Detection.

Defines model parameters, class definitions, inference thresholds, paths,
and runtime settings for edge deployment on Raspberry Pi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Assumes this file is located in <project_root>/src/config.py.
    """
    return Path(__file__).resolve().parents[1]


# Standard class index to human-readable label mapping
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Enclosure",
    1: "Enclosure_Bottom",
    2: "Enclosure_Top",
}

# Distinct BGR color palette for visualization (OpenCV BGR order)
# Enclosure: Cyan/Teal (255, 178, 50)
# Enclosure_Bottom: Emerald Green (50, 205, 50)
# Enclosure_Top: Amber/Orange (0, 140, 255)
DEFAULT_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 178, 50),
    1: (50, 205, 50),
    2: (0, 140, 255),
}


@dataclass
class DetectorConfig:
    """Configuration container for MQ-4 YOLOv8n detector."""

    # Model checkpoint path (relative to project root or absolute)
    model_path: Path = field(
        default_factory=lambda: get_project_root() / "models" / "best.pt"
    )

    # Expected class names
    class_names: Dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_NAMES)
    )

    # Class colors for bounding box visualization
    class_colors: Dict[int, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_COLORS)
    )

    # Inference hyperparameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640  # Default input resolution: 640x640

    # Device configuration ('cpu' for Raspberry Pi CPU execution)
    device: str = "cpu"

    # CPU threading optimization (match Raspberry Pi 4-core Cortex-A76)
    num_threads: int = 4

    # Default camera settings
    camera_device_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Output directories
    predictions_dir: Path = field(
        default_factory=lambda: get_project_root() / "results" / "predictions"
    )
    performance_dir: Path = field(
        default_factory=lambda: get_project_root() / "results" / "performance"
    )

    def __post_init__(self) -> None:
        """Resolve paths, check environment overrides, and ensure directories exist."""
        # Allow environment variable overrides
        env_model = os.getenv("MQ4_MODEL_PATH")
        if env_model:
            self.model_path = Path(env_model)

        env_conf = os.getenv("MQ4_CONF_THRESH")
        if env_conf:
            try:
                self.confidence_threshold = float(env_conf)
            except ValueError:
                pass

        env_iou = os.getenv("MQ4_IOU_THRESH")
        if env_iou:
            try:
                self.iou_threshold = float(env_iou)
            except ValueError:
                pass

        env_imgsz = os.getenv("MQ4_IMAGE_SIZE")
        if env_imgsz:
            try:
                self.image_size = int(env_imgsz)
            except ValueError:
                pass

        # Ensure model_path is absolute
        if not self.model_path.is_absolute():
            self.model_path = (get_project_root() / self.model_path).resolve()

        # Ensure output directories are absolute and exist
        if not self.predictions_dir.is_absolute():
            self.predictions_dir = (get_project_root() / self.predictions_dir).resolve()
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        if not self.performance_dir.is_absolute():
            self.performance_dir = (get_project_root() / self.performance_dir).resolve()
        self.performance_dir.mkdir(parents=True, exist_ok=True)

    def validate_checkpoint(self) -> Path:
        """Verify that the model checkpoint exists and is non-empty.

        Returns:
            Resolved Path to the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
            ValueError: If the checkpoint file is zero-byte / corrupt.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: '{self.model_path}'.\n"
                f"Please ensure 'best.pt' is placed in the models directory "
                f"({get_project_root() / 'models'}) or specify MQ4_MODEL_PATH."
            )

        if self.model_path.stat().st_size == 0:
            raise ValueError(
                f"Model checkpoint file at '{self.model_path}' is empty (0 bytes). "
                f"Please verify file integrity."
            )

        return self.model_path

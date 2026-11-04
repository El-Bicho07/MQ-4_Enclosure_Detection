"""Unit and integration test suite for MQ-4 Enclosure Detection.

Verifies model checkpoint existence, model loading, class labeling,
image inference execution, output validation, assembly status logic,
V4L2 hardware queries, Active Cooler telemetry, and camera safety.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.camera import (
    ThreadedCamera,
    compute_assembly_status,
    compute_box_iou,
    get_v4l2_device_info,
    list_available_cameras,
    run_camera_stream,
    test_camera_device as check_camera_device,
)
from src.config import DEFAULT_CLASS_NAMES, DetectorConfig, get_project_root
from src.detector import DetectionResult, MQ4Detector
from src.image_inference import run_image_inference
from src.utils import (
    create_synthetic_test_image,
    get_active_cooler_fan_rpm,
    get_soc_temperature,
    get_throttled_status,
    is_active_cooler_tachometer_available,
    validate_active_cooler_health,
)


class TestMQ4EnclosureDetector(unittest.TestCase):
    """Test suite for MQ-4 YOLOv8n object detection system."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment and initialize detector instance."""
        cls.config = DetectorConfig()
        cls.project_root = get_project_root()
        cls.test_dir = cls.project_root / "tests" / "test_artifacts"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Generate a synthetic test image for inference tests
        cls.test_image_path = cls.test_dir / "sample_enclosure_test.jpg"
        cls.test_image_np = create_synthetic_test_image(
            width=640, height=640, save_path=cls.test_image_path
        )

        # Load detector once for class to optimize test execution time
        cls.detector = MQ4Detector(config=cls.config)

    def test_01_checkpoint_exists(self) -> None:
        """Verify that the model checkpoint file exists and is non-empty."""
        checkpoint_path = self.config.model_path
        self.assertTrue(
            checkpoint_path.exists(),
            f"Checkpoint file not found at: {checkpoint_path}",
        )
        self.assertTrue(
            checkpoint_path.is_file(),
            f"Checkpoint path is not a file: {checkpoint_path}",
        )
        file_size_bytes = checkpoint_path.stat().st_size
        self.assertGreater(
            file_size_bytes,
            1_000_000,
            f"Checkpoint file appears too small ({file_size_bytes} bytes).",
        )

    def test_02_config_validation(self) -> None:
        """Test configuration validation and path resolution."""
        valid_path = self.config.validate_checkpoint()
        self.assertEqual(valid_path, self.config.model_path)

        # Test missing checkpoint raises FileNotFoundError
        bogus_config = DetectorConfig(model_path=Path("non_existent_model_weights.pt"))
        with self.assertRaises(FileNotFoundError):
            bogus_config.validate_checkpoint()

    def test_03_model_loads_successfully(self) -> None:
        """Verify that MQ4Detector loads the YOLO model weights without error."""
        self.assertIsNotNone(self.detector.model)
        self.assertTrue(hasattr(self.detector.model, "predict"))

    def test_04_three_expected_classes(self) -> None:
        """Verify that the model contains exactly the three expected classes:

        ID 0: Enclosure
        ID 1: Enclosure_Bottom
        ID 2: Enclosure_Top
        """
        model_names = self.detector.model.names
        self.assertIsNotNone(model_names, "Model has no 'names' dictionary.")
        self.assertEqual(
            len(model_names),
            3,
            f"Expected exactly 3 classes, but found {len(model_names)}: {model_names}",
        )

        for class_id, expected_label in DEFAULT_CLASS_NAMES.items():
            self.assertIn(
                class_id,
                model_names,
                f"Class ID {class_id} missing from model classes.",
            )
            self.assertEqual(
                model_names[class_id],
                expected_label,
                f"Class ID {class_id} mismatch: expected '{expected_label}', got '{model_names[class_id]}'",
            )

    def test_05_synthetic_image_inference(self) -> None:
        """Verify that image inference runs successfully on an input image."""
        result = self.detector.predict(
            source=self.test_image_np,
            conf=0.1,  # Low threshold to capture any features in synthetic image
            imgsz=320,  # Fast inference resolution
        )
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.boxes, list)
        self.assertIsInstance(result.scores, list)
        self.assertIsInstance(result.class_ids, list)
        self.assertIsInstance(result.class_names, list)
        self.assertIsInstance(result.speed, dict)
        self.assertIn("inference", result.speed)
        self.assertGreater(result.speed["inference"], 0.0)

    def test_06_detection_output_validity(self) -> None:
        """Verify bounding box coordinates and confidence score ranges."""
        result = self.detector.predict(
            source=self.test_image_path,
            conf=0.05,
            imgsz=320,
        )

        self.assertEqual(len(result.boxes), len(result.scores))
        self.assertEqual(len(result.boxes), len(result.class_ids))
        self.assertEqual(len(result.boxes), len(result.class_names))

        for box, score, cls_id, cls_name in zip(
            result.boxes, result.scores, result.class_ids, result.class_names
        ):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertIn(cls_id, [0, 1, 2])
            self.assertIn(cls_name, ["Enclosure", "Enclosure_Bottom", "Enclosure_Top"])
            self.assertEqual(len(box), 4)
            x1, y1, x2, y2 = box
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)

    def test_07_image_inference_pipeline_and_file_save(self) -> None:
        """Verify the full run_image_inference pipeline including saving output."""
        pred_dir = self.test_dir / "predictions"
        result, saved_file, annotated_img = run_image_inference(
            image_path=self.test_image_path,
            detector=self.detector,
            config=self.config,
            conf_thresh=0.25,
            imgsz=320,
            output_dir=pred_dir,
            save_annotated=True,
            show_window=False,
        )
        self.assertIsInstance(result, DetectionResult)
        self.assertIsNotNone(saved_file)
        self.assertTrue(saved_file.exists())
        self.assertGreater(saved_file.stat().st_size, 0)
        self.assertEqual(annotated_img.shape[:2], (640, 640))

    def test_08_invalid_image_inputs_handled_gracefully(self) -> None:
        """Verify detector handles invalid or nonexistent input sources gracefully."""
        with self.assertRaises(FileNotFoundError):
            self.detector.predict("nonexistent_dummy_image.jpg")
        with self.assertRaises(ValueError):
            self.detector.predict(None)
        with self.assertRaises(ValueError):
            self.detector.predict(np.array([]))

    def test_09_camera_initialization_failure_handling(self) -> None:
        """Verify camera stream gracefully handles non-existent camera devices."""
        with self.assertRaises(RuntimeError) as ctx:
            run_camera_stream(
                device=999,
                detector=self.detector,
                config=self.config,
                headless=True,
                max_frames=1,
            )
        self.assertIn("Failed to initialize camera source", str(ctx.exception))
        self.assertIn("Troubleshooting Steps for Raspberry Pi", str(ctx.exception))

    def test_10_annotation_rendering(self) -> None:
        """Verify that draw_annotations executes and returns a valid image."""
        dummy_result = DetectionResult(
            boxes=[[50.0, 50.0, 200.0, 200.0], [250.0, 250.0, 450.0, 450.0]],
            scores=[0.95, 0.88],
            class_ids=[0, 1],
            class_names=["Enclosure", "Enclosure_Bottom"],
            speed={"preprocess": 2.5, "inference": 180.0, "postprocess": 1.5},
            orig_shape=(640, 640),
        )
        annotated = self.detector.draw_annotations(
            self.test_image_np, dummy_result, show_fps=True
        )
        self.assertEqual(annotated.shape, self.test_image_np.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_11_list_available_cameras(self) -> None:
        """Verify camera enumeration executes safely without hanging or raising exceptions."""
        devices = list_available_cameras()
        self.assertIsInstance(devices, list)
        # When Logitech C270 is connected, device_index 0 should be present
        if devices:
            self.assertEqual(devices[0]["device_index"], 0)
            self.assertIn("C270", devices[0]["card_name"])

    def test_12_camera_diagnostic_failure(self) -> None:
        """Verify test_camera_device returns failure cleanly for nonexistent device 999."""
        ok, msg, path = check_camera_device(device=999, save_test_image=False)
        self.assertFalse(ok)
        self.assertIn("could not be opened", msg)
        self.assertIsNone(path)

    def test_13_compute_assembly_status(self) -> None:
        """Verify quality control status computation across assembly permutations."""
        # Full enclosure assembly
        res_full = DetectionResult(boxes=[[10, 10, 100, 100]], scores=[0.95], class_ids=[0], class_names=["Enclosure"])
        text, color, detail = compute_assembly_status(res_full)
        self.assertIn("PASS: ASSEMBLED ENCLOSURE", text)

        # Base + Cap spatially aligned
        res_aligned = DetectionResult(
            boxes=[[50, 100, 200, 250], [55, 80, 195, 240]],
            scores=[0.9, 0.88],
            class_ids=[1, 2],
            class_names=["Enclosure_Bottom", "Enclosure_Top"],
        )
        text, color, detail = compute_assembly_status(res_aligned)
        self.assertIn("CANDIDATE ASSEMBLY", text)

        # Base + Cap unaligned / detached
        res_detached = DetectionResult(
            boxes=[[10, 10, 50, 50], [400, 400, 500, 500]],
            scores=[0.9, 0.88],
            class_ids=[1, 2],
            class_names=["Enclosure_Bottom", "Enclosure_Top"],
        )
        text, color, detail = compute_assembly_status(res_detached)
        self.assertIn("DETACHED COMPONENTS", text)

        # Empty
        res_empty = DetectionResult()
        text, color, detail = compute_assembly_status(res_empty)
        self.assertIn("READY: NO ENCLOSURE DETECTED", text)

    def test_14_threaded_camera_failure(self) -> None:
        """Verify ThreadedCamera raises RuntimeError when opening invalid device."""
        with self.assertRaises(RuntimeError):
            ThreadedCamera(src=999)

    def test_15_v4l2_device_query_on_hardware(self) -> None:
        """Verify POSIX ioctl query correctly identifies video0 and skips invalid paths."""
        info = get_v4l2_device_info("/dev/video0")
        if info is not None:
            self.assertTrue(info["is_capture"])
            self.assertIn("C270", info["card"])

        # /dev/video1 is metadata capture node, should return None
        info_meta = get_v4l2_device_info("/dev/video1")
        self.assertIsNone(info_meta)

        # Nonexistent device should return None
        info_bogus = get_v4l2_device_info("/dev/video999")
        self.assertIsNone(info_bogus)

    def test_16_box_iou_computation(self) -> None:
        """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
        box_a = [10.0, 10.0, 100.0, 100.0]
        box_b = [10.0, 10.0, 100.0, 100.0]
        box_c = [55.0, 10.0, 145.0, 100.0]
        box_d = [300.0, 300.0, 400.0, 400.0]

        self.assertAlmostEqual(compute_box_iou(box_a, box_b), 1.0)
        self.assertGreater(compute_box_iou(box_a, box_c), 0.2)
        self.assertEqual(compute_box_iou(box_a, box_d), 0.0)

    def test_17_active_cooler_and_soc_telemetry(self) -> None:
        """Verify SoC temperature and Active Cooler tachometer interfaces.

        Validates temperature readability within plausible thermal limits,
        and distinguishes between:
          1. Tachometer interface unavailable (None).
          2. Tachometer interface available with fan stopped/idle (0 RPM).
          3. Tachometer interface available with fan spinning (> 0 RPM).

        Zero RPM is accepted as a valid idle state when cooling is not required.
        Negative or invalid RPM values are rejected.
        """
        from unittest.mock import patch

        # 1. Verify SoC temperature readability and plausible operating range
        temp = get_soc_temperature()
        self.assertIsNotNone(temp, "SoC temperature must be readable via sysfs")
        self.assertIsInstance(temp, float)
        self.assertGreater(temp, 15.0, f"Plausibility failure: temp {temp}°C is abnormally low")
        self.assertLess(temp, 95.0, f"Plausibility failure: temp {temp}°C exceeds max operating range")

        # 2. Check tachometer interface availability vs reading
        is_tach_avail = is_active_cooler_tachometer_available()
        fan_rpm = get_active_cooler_fan_rpm()

        if is_tach_avail:
            # Tachometer interface exists on hardware platform
            self.assertIsNotNone(
                fan_rpm,
                "Tachometer sysfs node is detected but get_active_cooler_fan_rpm() returned None",
            )
            self.assertIsInstance(fan_rpm, int, "Fan RPM must be an integer")
            self.assertGreaterEqual(fan_rpm, 0, f"Fan RPM cannot be negative: {fan_rpm}")
            self.assertLess(fan_rpm, 15000, f"Fan RPM exceeds physical tachometer ceiling: {fan_rpm}")

            # Distinguish idle vs active spinning states
            if fan_rpm == 0:
                # Valid idle state: Active Cooler PWM governor shuts off fan when SoC is cool
                self.assertEqual(fan_rpm, 0)
            else:
                # Valid active state: Fan actively providing forced convection cooling
                self.assertGreater(fan_rpm, 0)
        else:
            # Platform without Active Cooler tachometer (e.g. passive heatsink, VM, non-Pi)
            self.assertIsNone(
                fan_rpm,
                "Expected None for fan RPM on system without Active Cooler tachometer interface",
            )

        # 3. Verify rejection of negative or corrupted tachometer readings
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="-100\n"):
                self.assertIsNone(
                    get_active_cooler_fan_rpm(),
                    "Negative RPM readings must be rejected as invalid and return None",
                )
            with patch("pathlib.Path.read_text", return_value="not_a_number\n"):
                self.assertIsNone(
                    get_active_cooler_fan_rpm(),
                    "Non-numeric tachometer readings must return None without raising unhandled exceptions",
                )

        # 4. Verify explicit state handling under controlled scenarios:
        # 4a. Tachometer available and reporting zero RPM (valid idle state)
        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value="0\n"):
            rpm_zero = get_active_cooler_fan_rpm()
            self.assertEqual(rpm_zero, 0, "Idle state (0 RPM) must be accepted as valid integer 0")

        # 4b. Tachometer available and reporting positive RPM (active cooling state)
        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", return_value="3100\n"):
            rpm_active = get_active_cooler_fan_rpm()
            self.assertEqual(rpm_active, 3100, "Active cooling state must return positive RPM")

        # 4c. Tachometer unavailable (returns None)
        with patch("pathlib.Path.exists", return_value=False):
            rpm_none = get_active_cooler_fan_rpm()
            self.assertIsNone(rpm_none, "Unavailable interface must return None without raising an exception")

        # 5. Verify firmware throttling telemetry
        throttled = get_throttled_status()
        self.assertIsInstance(throttled, dict)
        self.assertIn("is_currently_throttled", throttled)
        self.assertIn("code_hex", throttled)
        self.assertIn("is_under_voltage", throttled)
        self.assertIn("is_freq_capped", throttled)

    def test_18_multi_enclosure_assembly_status(self) -> None:
        """Verify quality status handles multiple full enclosure assemblies in one frame."""
        res_multi = DetectionResult(
            boxes=[[10, 10, 100, 100], [200, 200, 300, 300]],
            scores=[0.95, 0.92],
            class_ids=[0, 0],
            class_names=["Enclosure", "Enclosure"],
        )
        text, color, detail = compute_assembly_status(res_multi)
        self.assertIn("PASS: 2 ASSEMBLED ENCLOSURES", text)
        self.assertEqual(color, (50, 205, 50))
        self.assertIn("2D optical vision cannot verify", detail)

    def test_19_ambiguous_and_detached_assembly_permutations(self) -> None:
        """Verify handling of mixed scenes with complete units + detached or orphaned parts."""
        # Mixed: 1 full enclosure + 1 loose bottom
        res_mixed = DetectionResult(
            boxes=[[10, 10, 100, 100], [400, 400, 500, 500]],
            scores=[0.95, 0.85],
            class_ids=[0, 1],
            class_names=["Enclosure", "Enclosure_Bottom"],
        )
        text, color, detail = compute_assembly_status(res_mixed)
        self.assertIn("CANDIDATE ASSEMBLY", text)
        self.assertIn("Mixed scene", detail)

        # 2 bottoms + 1 top (1 candidate pair + 1 orphaned bottom)
        res_unpaired = DetectionResult(
            boxes=[[50, 100, 200, 250], [55, 80, 195, 240], [500, 500, 600, 600]],
            scores=[0.9, 0.88, 0.75],
            class_ids=[1, 2, 1],
            class_names=["Enclosure_Bottom", "Enclosure_Top", "Enclosure_Bottom"],
        )
        text, color, detail = compute_assembly_status(res_unpaired)
        self.assertIn("DETACHED COMPONENTS", text)
        self.assertIn("unpaired base", detail)

    def test_20_spatial_pairing_prevents_cross_assembly_matching(self) -> None:
        """Verify spatial pairing matches closest components rather than crossing assemblies."""
        # Assembly A at x~50: Bottom and Top aligned
        # Assembly B at x~450: Bottom and Top aligned
        res_two_assemblies = DetectionResult(
            boxes=[
                [50, 100, 200, 250],    # Bottom A
                [55, 80, 195, 240],     # Top A
                [450, 100, 600, 250],   # Bottom B
                [455, 80, 595, 240],    # Top B
            ],
            scores=[0.85, 0.99, 0.98, 0.84],  # Top A highest, Bottom B highest
            class_ids=[1, 2, 1, 2],
            class_names=["Enclosure_Bottom", "Enclosure_Top", "Enclosure_Bottom", "Enclosure_Top"],
        )
        text, color, detail = compute_assembly_status(res_two_assemblies)
        self.assertIn("2 CANDIDATE ASSEMBLIES", text)
        self.assertEqual(color, (0, 215, 255))

    def test_21_threaded_camera_lifecycle_and_metrics(self) -> None:
        """Verify ThreadedCamera metrics, frame tracking, and idempotent release."""
        from unittest.mock import MagicMock, patch

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 30.0)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = ThreadedCamera(src=0)
            ret, frame, latency = cam.read()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))

            metrics = cam.get_metrics()
            self.assertIn("frames_captured", metrics)
            self.assertIn("frames_served", metrics)
            self.assertIn("frames_overwritten", metrics)
            self.assertGreaterEqual(metrics["frames_served"], 1)

            # Idempotent release
            cam.release()
            self.assertTrue(cam.stopped)
            cam.release()

    def test_22_active_cooler_health_validation(self) -> None:
        """Verify Active Cooler health status logic under various thermal/fan conditions."""
        # 1. Normal idle (cool SoC, fan stopped)
        idle_status = validate_active_cooler_health(temp_c=45.0, fan_rpm=0)
        self.assertEqual(idle_status["status"], "NORMAL_IDLE")
        self.assertTrue(idle_status["is_healthy"])

        # 2. Active cooling (warm SoC, fan spinning)
        active_status = validate_active_cooler_health(temp_c=58.0, fan_rpm=3100)
        self.assertEqual(active_status["status"], "ACTIVE_COOLING")
        self.assertTrue(active_status["is_healthy"])

        # 3. Stall warning (hot SoC, fan not spinning)
        stall_status = validate_active_cooler_health(temp_c=62.0, fan_rpm=0)
        self.assertEqual(stall_status["status"], "STALL_WARNING")
        self.assertFalse(stall_status["is_healthy"])

        # 4. Tachometer anomaly (negative RPM)
        anomaly_status = validate_active_cooler_health(temp_c=50.0, fan_rpm=-50)
        self.assertEqual(anomaly_status["status"], "ANOMALY")
        self.assertFalse(anomaly_status["is_healthy"])

        # 5. Interface unavailable
        unavail_status = validate_active_cooler_health(temp_c=50.0, fan_rpm=None)
        self.assertEqual(unavail_status["status"], "UNAVAILABLE")
        self.assertTrue(unavail_status["is_healthy"])

    def test_23_box_iou_edge_cases(self) -> None:
        """Verify box IoU mathematical edge cases (zero area, disjoint, out-of-order)."""
        box1 = [10.0, 10.0, 100.0, 100.0]
        # Zero area box (degenerate width/height)
        box_zero = [50.0, 50.0, 50.0, 50.0]
        self.assertEqual(compute_box_iou(box1, box_zero), 0.0)

        # Disjoint boxes
        box_far = [500.0, 500.0, 600.0, 600.0]
        self.assertEqual(compute_box_iou(box1, box_far), 0.0)

        # Inverted box coordinates (x2 < x1)
        box_inverted = [100.0, 100.0, 10.0, 10.0]
        self.assertEqual(compute_box_iou(box1, box_inverted), 0.0)

    def test_24_benchmark_dual_fps_calculation(self) -> None:
        """Verify benchmark result structure contains both throughput FPS and reciprocal FPS."""
        from src.utils import ResolutionBenchmarkResult

        res = ResolutionBenchmarkResult(
            imgsz=320,
            num_runs=30,
            preprocess_mean_ms=1.81,
            inference_mean_ms=279.77,
            postprocess_mean_ms=1.31,
            total_latency_mean_ms=283.65,
            total_latency_std_ms=15.2,
            total_latency_p50_ms=281.0,
            total_latency_p95_ms=369.90,
            total_latency_min_ms=270.0,
            total_latency_max_ms=410.0,
            average_fps=3.52,
            cpu_percent_during_run=68.3,
            ram_used_mb=2957.0,
            reciprocal_fps=3.53,
            throughput_fps=3.52,
            warmup_runs=5,
        )
        self.assertEqual(res.throughput_fps, 3.52)
        self.assertEqual(res.reciprocal_fps, 3.53)
        self.assertEqual(res.warmup_runs, 5)

    def test_25_model_provenance_and_metadata(self) -> None:
        """Verify that models/best.pt contains authentic training metadata and class IDs."""
        import torch

        checkpoint_path = self.project_root / "models" / "best.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Verify class names dictionary
        expected_names = {0: "Enclosure", 1: "Enclosure_Bottom", 2: "Enclosure_Top"}
        model_names = ckpt.get("model").names if hasattr(ckpt.get("model"), "names") else None
        self.assertEqual(model_names, expected_names)

        # Verify training arguments
        train_args = ckpt.get("train_args")
        self.assertIsNotNone(train_args)
        self.assertEqual(train_args.get("model"), "yolov8n.pt")
        self.assertEqual(train_args.get("imgsz"), 640)
        self.assertIn("MQ-4", train_args.get("data", ""))


if __name__ == "__main__":
    unittest.main()

# Assessment Evidence: Logitech C270 & Active Cooler Hardware Validation

## 1. Hardware Audit & Device Identification

A systematic audit was conducted on the physical camera and cooling hardware attached to the Raspberry Pi 5.

### USB Peripheral Identification (`lsusb`):
```text
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 30fa:1040 INSTANT USB GAMING MOUSE 
Bus 001 Device 003: ID 046d:0825 Logitech, Inc. Webcam C270
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 003 Device 002: ID 413c:2113 Dell Computer Corp. KB216 Wired Keyboard
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
```
- **Manufacturer:** Logitech, Inc.
- **Model:** Logitech Webcam C270
- **USB Vendor ID:** `046d`
- **USB Product ID:** `0825`
- **USB Bus / Address:** Bus 001, Device 003

### Video Node Topology & Capabilities:

The Linux UVC driver (`uvcvideo`) exposes **two distinct device nodes** for this single USB camera:

| Device Node | Subsystem Name | V4L2 Capabilities | Role |
| :--- | :--- | :--- | :--- |
| **`/dev/video0`** | `C270 HD WEBCAM` | `0x04200001` (Video Capture, Streaming, Extended Pix Format) | **Primary Video Stream** (Used by application) |
| **`/dev/video1`** | `C270 HD WEBCAM` | `0x04a00000` (Metadata Capture, Streaming) | **UVC Payload Header Metadata** (Not a camera) |

> [!IMPORTANT]
> The detection codebase utilizes POSIX `ioctl(VIDIOC_QUERYCAP)` to inspect `effective_caps & V4L2_CAP_VIDEO_CAPTURE (0x00000001)`. This immediately rejects `/dev/video1` and internal Raspberry Pi 5 ISP (`pisp_be`) and decoder (`hevc`) nodes (`/dev/video19`–`video35`), eliminating 10-second `select()` timeouts during camera discovery.

---

## 2. Supported Formats, Resolutions, and Frame Rates

Queried via `v4l2-ctl --list-formats-ext -d /dev/video0`:

### A. Raw Uncompressed (`YUYV` 4:2:2)
- `640 × 480` @ 30, 24, 20, 15, 10, 7.5, 5 FPS
- `320 × 240` @ 30, 24, 20, 15, 10, 7.5, 5 FPS
- `1280 × 720` (720p HD) @ **10 FPS max** (Bandwidth-constrained by USB 2.0)

### B. Compressed Motion-JPEG (`MJPG`)
- `640 × 480` @ 30, 24, 20, 15, 10, 7.5, 5 FPS
- `800 × 600` @ 30, 24, 20, 15, 10, 7.5, 5 FPS
- `1280 × 720` (720p HD) @ **30 FPS** (Hardware JPEG encoder active in camera)

**Implementation Choice:** `src/camera.py` sets `cv2.CAP_PROP_FOURCC = 'MJPG'` to leverage hardware encoding, enabling full 30 FPS sensor capture.

---

## 3. Camera Diagnostics Verification

### Device Auto-Discovery:
```bash
python3 src/camera.py --list-devices
```
**Output:**
```text
============================================================
SCANNING FOR CONNECTED CAMERA DEVICES...
============================================================
  Index: 0 | Path: /dev/video0 | Name: C270 HD WEBCAM | Format: 640x480 @ ~30 FPS | Capture OK: True
============================================================
```

### Isolated Hardware Test (Without YOLO Model Load):
```bash
python3 src/camera.py --test-camera --device 0
```
**Output:**
```text
============================================================
TESTING CAMERA DEVICE: 0
============================================================
Status:  SUCCESS
Details: Camera '0' is operational: Native 640x480 @ ~30.0 FPS. Captured frame shape: (480, 640, 3).
Frame:   Saved to /home/team5/MQ-4_Enclosure_Detection/results/predictions/camera_test_frame.jpg
============================================================
```

---

## 4. Raspberry Pi 5 Active Cooler Monitoring

The official Raspberry Pi 5 Active Cooler was verified operational. System monitoring interfaces:

- **Thermal Sensor Interface:** `/sys/class/thermal/thermal_zone0/temp` (SoC die temperature)
- **Fan Hardware Monitor:** `/sys/class/hwmon/hwmon2/fan1_input` (Actual tachometer RPM)
- **PWM Drive Level:** `/sys/class/hwmon/hwmon2/pwm1` (0 to 255)
- **Throttling Bitmask:** `vcgencmd get_throttled`

### Active Cooler Baseline State:
- **Baseline Temperature:** ~45.0 °C to 50.5 °C (Idle)
- **Fan Tachometer:** 0 RPM (Idle when < 50 °C) to ~2,800–3,200 RPM (when > 50 °C)
- **Throttling State:** `throttled=0x0` (No under-voltage, no frequency capping, no thermal throttling)

---

## 5. Active Cooler Operating States & Fan Telemetry Validation

### A. Temperature-Controlled Governor Behavior
The Raspberry Pi 5 Active Cooler uses a 4-pin PWM-driven fan with tachometer feedback managed automatically by the kernel `pwmfan` driver (`/sys/devices/platform/cooling_fan/hwmon/hwmon2/`). The default firmware governor defines progressive thermal trip points:

| Cooling State | SoC Temp Range | Target Fan Speed | Operational Behavior |
| :---: | :---: | :---: | :--- |
| **State 0** | `< 50.0 °C` | **0 RPM** | **Idle / Fan Stopped** (Acoustically silent; valid normal operating state) |
| **State 1** | `50.0 – 60.0 °C` | ~2,800 – 3,200 RPM | Low-speed continuous convection cooling |
| **State 2** | `60.0 – 67.5 °C` | ~4,800 RPM | Moderate active cooling under sustained edge inference |
| **State 3** | `67.5 – 75.0 °C` | ~6,000 RPM | High-speed cooling during extreme multi-threaded workloads |
| **State 4** | `> 75.0 °C` | ~8,000 RPM | Maximum emergency cooling before 80 °C thermal throttle |

> [!NOTE]
> **Telemetry Validation vs. Motor Operation:**
> Reading `0 RPM` from `/sys/class/hwmon/hwmon2/fan1_input` is the expected, correct reading when the SoC temperature is below 50.0 °C. The unit test verifies interface readability, plausible numerical ranges, and rejection of negative values—it does **not** assume the fan must spin constantly.

### B. Non-Invasive Fan-Operation Validation Procedure
To verify that the physical fan motor and PWM drive circuit engage without forcing settings or artificially stressing the hardware:
1. Note idle baseline temperature: `python3 -c "from src.utils import get_soc_temperature; print(get_soc_temperature())"` (~45–48 °C, Fan: 0 RPM).
2. Execute a standard multi-resolution benchmark or continuous inference run:
   ```bash
   .venv/bin/python3 src/utils.py --benchmark --runs 25
   ```
3. During this CPU vision workload, the SoC temperature will naturally rise above 50.0 °C.
4. Confirm fan spin-up via tachometer:
   ```bash
   python3 -c "from src.utils import get_active_cooler_fan_rpm, get_soc_temperature; print(f'Temp: {get_soc_temperature()} C, Fan: {get_active_cooler_fan_rpm()} RPM')"
   ```
   *Expected output: `Temp: 54.2 C, Fan: 2874 RPM` (or higher).*
5. No edits to `/boot/firmware/config.txt` or manual sysfs override are required or permitted.

---

## 6. Assembly Quality Status Validation Methodology

The function `compute_assembly_status()` was analyzed for quality control reliability:

### Identified Risk:
A naive rule declaring an enclosure assembled merely because Class 0 (`Enclosure`), Class 1 (`Enclosure_Bottom`), or Class 2 (`Enclosure_Top`) exist anywhere in the image will trigger false positives if parts are laying loose on a conveyor.

### Implemented Spatial Verification:
1. **Monolithic Unit:** If Class 0 (`Enclosure`) is detected with high confidence (≥ 0.25), it represents the completed mated housing.
2. **Sub-Component Co-Presence & IoU:** When Class 1 (`Enclosure_Bottom`) and Class 2 (`Enclosure_Top`) both appear:
   - Calculate Bounding Box IoU (`compute_box_iou`).
   - If `IoU > 0.15` and top vertical center is above bottom center (`t_center_y < b_center_y`), flag as `NOTICE: CANDIDATE ASSEMBLY (VERIFY LATCH)`.
   - If spatially disjoint (`IoU <= 0.05`), flag as `WARN: DETACHED COMPONENTS IN FRAME`.

### Physical Engineering Disclaimer:
> [!CAUTION]
> **2D Optical Limitation:** While YOLO detects visual component presence and spatial overlap, it **cannot** determine physical snap-fit latching force, seal gasket compression, or internal screw torque. Physical assembly verification stations must combine computer vision with mechanical tactile seating sensors or optical displacement depth sensors.


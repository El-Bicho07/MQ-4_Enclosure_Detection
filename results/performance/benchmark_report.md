# MQ-4 Enclosure Detection - Performance Benchmark Report

**Generated:** 2026-11-04 07:17:50 UTC

## 1. Hardware & System Telemetry

- **Platform:** `Linux-6.18.50+rpt-rpi-2712-aarch64-with-glibc2.41`
- **Architecture:** `aarch64`
- **Logical CPUs:** `4`
- **Total RAM:** `4049.1 MB`
- **SoC Temperature:** `60.6 °C`
- **Model Load Latency:** `224.13 ms`

## 2. Training-Environment Reference Metrics (Google Colab)

> [!NOTE]
> The metrics below represent previously recorded evaluation results on the held-out test set (72 images) in the Colab training environment. They do NOT reflect Raspberry Pi inference speed.

| Metric | Colab Test Set Value |
| :--- | :--- |
| **Precision** | 97.8% |
| **Recall** | 98.7% |
| **mAP50** | 99.2% |
| **mAP50-95** | 74.8% |

## 3. Measured Raspberry Pi Edge Performance

> [!IMPORTANT]
> These are empirical, measured benchmarks collected directly on this device CPU. Numbers are not simulated or hardcoded.

| Resolution | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Mean Latency (ms) | P95 Latency (ms) | Average FPS | CPU % | RAM Used (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **320x320** | 1.81 | 279.77 | 1.31 | **283.65** | 369.9 | **3.52** | 68.3% | 2957.7 |
| **480x480** | 2.86 | 497.28 | 1.15 | **502.13** | 550.89 | **1.99** | 79.2% | 2939.5 |
| **640x640** | 3.32 | 813.39 | 1.2 | **818.82** | 858.95 | **1.22** | 82.5% | 3008.0 |

## 4. Key Takeaways & Edge Optimization Insights

- **Resolution Impact:** Reducing inference image size (e.g. from 640x640 to 480x480 or 320x320) substantially cuts latency and boosts FPS on CPU-constrained devices.
- **Threading Optimization:** Multi-threaded PyTorch CPU execution (`torch.set_num_threads(4)`) leverages all 4 ARM Cortex-A76 cores.
- **Memory Footprint:** YOLOv8n remains lightweight, consuming minimal RAM and avoiding memory pressure.

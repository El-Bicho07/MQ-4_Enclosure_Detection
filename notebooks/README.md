# Notebooks and Experimentation Workflows

## 1. Overview

This directory contains reference documentation and notebooks for:
- Initial model prototyping and Colab-to-edge migration.
- Dataset exploratory data analysis (EDA).
- Performance validation and post-training analysis.

## 2. Training Notebook (Google Colab Reference)

The baseline model checkpoint (`models/best.pt`) was trained in Google Colab utilizing standard cloud GPU infrastructure.

### Typical Colab Pipeline:

1. **Environment Setup:**
   ```bash
   pip install ultralytics
   ```

2. **Dataset Acquisition & Configuration:**
   - Dataset: `MQ-4_Dataset_v1_Baseline`
   - Formatted into `data_train.yaml` with 3 target classes.

3. **YOLOv8n Training Command:**
   ```python
   from ultralytics import YOLO

   # Initialize YOLOv8 nano model with pretrained COCO weights
   model = YOLO("yolov8n.pt")

   # Train on MQ-4 enclosure dataset
   results = model.train(
       data="data_train.yaml",
       epochs=100,
       imgsz=640,
       batch=16,
       device=0,  # NVIDIA GPU
       save=True,
       project="MQ-4_Sensor_Detection",
       name="colab_baseline_run",
   )
   ```

4. **Validation & Test Set Evaluation:**
   ```python
   # Run validation on held-out test split (72 images)
   metrics = model.val(data="data_train.yaml", split="test")
   print("Test Precision:", metrics.box.p)
   print("Test Recall:", metrics.box.r)
   print("Test mAP50:", metrics.box.map50)
   print("Test mAP50-95:", metrics.box.map)
   ```

5. **Weight Export:**
   - The best performing checkpoint (`best.pt`) was downloaded and placed into `models/best.pt` in this repository.

## 3. Local Edge Prototyping

For interactive inspection and debugging on the Raspberry Pi:
- Use interactive Python or launch Jupyter inside `.venv`:
  ```bash
  source .venv/bin/activate
  python3 -m pip install jupyterlab  # Optional
  jupyter lab --ip=0.0.0.0 --no-browser
  ```

# Assessment Evidence: Dataset Preparation

## 1. Dataset Origin & Provenance

- **Dataset Identifier:** `MQ-4_Dataset_v1_Baseline`
- **Application Scope:** Automated visual inspection and quality verification of 3D-printed/molded MQ-4 methane/natural gas sensor enclosures during assembly.
- **Source Environment:** Assembly workstation camera under varying illumination and orientation.
- **Annotation Tool:** Roboflow / LabelImg (YOLO annotation standard).

## 2. Partitioning & Balance

| Split | Count | Ratio | Class Representation |
| :--- | :---: | :---: | :--- |
| **Training** | 498 | 69.9% | Balanced across full assemblies, base plates, and caps |
| **Validation** | 142 | 19.9% | Stratified sample for early-stopping and checkpoint selection |
| **Test** | 72 | 10.1% | Completely isolated held-out test set for final metric reporting |
| **Total** | **712** | **100.0%** | Baseline inspection dataset |

## 3. Class Definitions & Annotation Rules

```
Class ID 0: Enclosure
  - Bounding box encompasses the fully mated enclosure assembly.
  - Sensor cylindrical top visible alongside base latching tabs.

Class ID 1: Enclosure_Bottom
  - Lower mounting chassis.
  - Rectangular footprint with mounting ear cutouts and terminal pin openings.

Class ID 2: Enclosure_Top
  - Upper protective dome.
  - Characteristic circular mesh vent and snap-fit side tabs.
```

## 4. Preprocessing & Augmentations (Training Environment)

Augmentations applied during dataset generation in training pipeline:
- **Spatial:** Random horizontal flip (p=0.5), random scaling (±15%), shear (±5°), rotation (±10°).
- **Photometric:** Hue adjustments (±15°), saturation (±25%), brightness (±20%).
- **Mosaic:** Ultralytics default 4-image mosaic composition enabled for training epochs 1–90, disabled for last 10 epochs.
- **Standardized Resolution:** 640 × 640 pixels (RGB).

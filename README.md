# Carbon Fibre Defect Semantic Segmentation with SegFormer

Deep learning pipeline for semantic segmentation of defects in carbon fibre composites using the SegFormer architecture.

This project was developed during an internship at **PROFACTOR GmbH** and focuses on detecting manufacturing defects from multi-channel industrial imaging data.

---

## Overview

Carbon fibre composite materials are widely used in aerospace and automotive industries.  
Automated defect detection is essential for quality control, but the complex texture of carbon fibre materials makes segmentation challenging.

This repository implements a training and evaluation pipeline based on **SegFormer** for detecting defects in carbon fibre materials using industrial imaging data.

The repository includes:

- SegFormer training pipeline
- dataset loading for FScan data
- evaluation scripts
- visualization utilities for segmentation results

---

## Example Results
---

### Segmentation comparison

| Input Image | Ground Truth | Prediction |
|-------------|--------------|------------|
| ![](assets/images/input_1.png) | ![](assets/images/gt_1.png) | ![](assets/images/pred_1.png) |
| ![](assets/images/input_2.png) | ![](assets/images/gt_2.png) | ![](assets/images/pred_2.png) |
| ![](assets/images/input_3.png) | ![](assets/images/gt_3.png) | ![](assets/images/pred_3.png) |
---

## Model Architecture

The segmentation model is based on **SegFormer**, a transformer-based architecture designed for efficient semantic segmentation.

Key components:

- Transformer encoder
- multi-scale feature fusion
- lightweight decoder head

---

## Dataset

The dataset used in this work is acquired using the **FScan sensor** developed at PROFACTOR.

⚠️ The original dataset **cannot be publicly released** because it contains proprietary industrial data.

However, the repository includes:

- dataset loading utilities
- preprocessing pipeline
- training configuration

These components can be adapted to other datasets.

---

## Training

Example training command:

```bash
python train_segformer.py \
    --config configs/segformer_config.yaml \
    --dataset_path path/to/fscan_dataset

# Change Image Detection Using Deep Learning (Sentinel-2)

## Overview

This project presents a deep learning-based semantic segmentation pipeline for **change detection** in land cover using **Sentinel-2 Level-1C** satellite imagery. The core of the approach is a **U-Net architecture**, trained on a **custom-labeled dataset** derived from Sentinel-2 data. Real-world application is demonstrated on satellite images from **Ahmedabad, India**, showcasing the model’s potential for monitoring land use and environmental change.

---

## Objectives

* Develop a U-Net segmentation model tailored for detecting changes in satellite images.
* Construct a labeled dataset from Sentinel-2 L1C imagery.
* Validate the model through inference on real, time-stamped Sentinel-2 imagery.

---

## Dataset Description

### Source

All data were acquired from the **Sentinel-2** Earth observation mission, part of the **European Space Agency (ESA)** Copernicus Programme.

### Custom Dataset Construction

* **Data Type:** Sentinel-2 Level-1C (L1C) top-of-atmosphere reflectance products
* **Band Selection:** Use of multi-spectral bands with 10m and 20m resolution
* **Processing Workflow:**

  * Selection of temporal image pairs for change detection
  * Normalization and resizing of bands
  * Label mask creation (manual/GIS-guided)
* **Application:** The resulting dataset enables supervised learning for detecting land cover changes between two image states.

---

## Model Architecture

### U-Net

* **Encoder:** Down-sampling path with convolutions and max-pooling
* **Decoder:** Up-sampling with transposed convolutions and skip connections
* **Output:** Pixel-wise segmentation mask indicating change regions
* **Loss Function:** Binary Cross-Entropy or Dice Loss
* **Framework:** PyTorch

### Training Configuration

| Parameter        | Value           |
| ---------------- | --------------- |
| Epochs           | 50              |
| Batch Size       | 16              |
| Optimizer        | Adam            |
| Learning Rate    | 0.001           |
| Image Size       | 128 × 128       |
| Loss Function    | BCE / Dice Loss |
| Validation Split | 20%             |

---

## Repository Structure

* [`model.ipynb`](https://github.com/DhrumilPrajapati03/Segmentation_model_using_U-net/blob/main/model.ipynb): Model training on the custom dataset built from Sentinel-2 imagery.
* [`Ahmedabad_sentinel_2.ipynb`](https://github.com/DhrumilPrajapati03/Segmentation_model_using_U-net/blob/main/Ahmedabad_sentinel_2.ipynb): Real-world inference using Sentinel-2 data for Ahmedabad to detect changes.

---

## Sample Outputs

| Sentinel-2 RGB Image   | Ground Truth Mask  | Predicted Change Mask  |
| ---------------------- | ------------------ | ---------------------- |
| ![RGB](path/to/image1) | ![GT](path/to/gt1) | ![Pred](path/to/pred1) |

> *Note: Add image paths or upload examples to visualize results.*

---

## Installation

### Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.11
* torchvision
* NumPy
* Matplotlib
* rasterio
* scikit-learn

### Setup

```bash
git clone https://github.com/DhrumilPrajapati03/Segmentation_model_using_U-net.git
cd Segmentation_model_using_U-net
pip install -r requirements.txt
```

> *Consider creating a `requirements.txt` file for reproducibility.*

---

## How to Use

1. **Train the Model:**
   Open `model.ipynb` and run all cells to train the segmentation model.

2. **Run Inference:**
   Open `Ahmedabad_sentinel_2.ipynb` to apply the trained model on new Sentinel-2 imagery and visualize changes.

---

## Acknowledgements

* **Data Source:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
* **Satellite Imagery:** Sentinel-2 Level-1C (L1C)
* **Architecture:** Inspired by the U-Net model proposed by Ronneberger et al. (2015)

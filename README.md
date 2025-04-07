# Segmentation_model_using_U-net
Your project involves change detection using a segmentation model (U-Net with a ResNet-34 backbone pre-trained on ImageNet) and transitioning from the OSCD (Onera Satellite Change Detection) dataset to your custom Sentinel-2 dataset. Since you’ve already preprocessed your real-world data (RGB bands from Sentinel-2, clipped and converted to GeoTIFF via QGIS), I’ll guide you step-by-step on how to test your existing model on this new dataset in a structured and detailed manner. I’ll assume you’re working in a Python environment (e.g., Jupyter Notebook, as per your GitHub repo) with libraries like PyTorch, Rasterio, and NumPy.

Here’s the detailed plan:

---

### Step 1: Understand Your Existing Model and Requirements
- **Model Recap:** Your GitHub repo shows a U-Net model with a ResNet-34 backbone, originally trained on the OSCD dataset for binary change detection (change vs. no-change masks). The input is likely RGB images (3 channels), and the output is a segmentation mask.
- **New Dataset:** You have two Sentinel-2 RGB GeoTIFFs (pre- and post-event: 22/02/2023 and 12/02/2025) representing your area of interest (AOI).
- **Goal:** Use your pre-trained model to predict change masks on this new dataset and evaluate its performance.

#### Action:
- Confirm your model’s input requirements (e.g., image size, normalization). From your repo, it seems you resized OSCD images to 256x256 and normalized them (likely using ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- Ensure your Sentinel-2 RGB data aligns with this (3 channels, compatible resolution).

---

### Step 2: Preprocess Your Custom Sentinel-2 Dataset
Your GeoTIFFs need to be prepared to match the model’s expected input format.

#### 2.1 Load and Inspect GeoTIFFs
- Use `rasterio` to read the RGB bands from your GeoTIFFs.
- Verify dimensions, data type, and coordinate reference system (CRS).

```python
import rasterio
import numpy as np

# Paths to your GeoTIFFs
pre_image_path = "path/to/2023-02-22_rgb.tif"
post_image_path = "path/to/2025-02-12_rgb.tif"

# Load images
with rasterio.open(pre_image_path) as pre_src:
    pre_image = pre_src.read([1, 2, 3])  # RGB bands (assuming R=1, G=2, B=3)
    pre_meta = pre_src.meta

with rasterio.open(post_image_path) as post_src:
    post_image = post_src.read([1, 2, 3])  # RGB bands
    post_meta = post_src.meta

# Transpose to (H, W, C) for easier handling
pre_image = np.transpose(pre_image, (1, 2, 0))
post_image = np.transpose(post_image, (1, 2, 0))

print("Pre-image shape:", pre_image.shape)
print("Post-image shape:", post_image.shape)
print("Data type:", pre_image.dtype)
```

#### 2.2 Normalize and Scale Pixel Values
- Sentinel-2 data is typically in 12-bit format (0–4095 or higher, depending on processing level). Your model expects 8-bit RGB (0–255) normalized to `[0, 1]` or ImageNet stats.
- Clip and scale the values.

```python
# Clip outliers (e.g., 0–4000 for Sentinel-2 Level-2A) and scale to 0–255
pre_image = np.clip(pre_image, 0, 4000) / 4000 * 255
post_image = np.clip(post_image, 0, 4000) / 4000 * 255

# Convert to uint8
pre_image = pre_image.astype(np.uint8)
post_image = post_image.astype(np.uint8)
```

#### 2.3 Tile Images (If Necessary)
- If your AOI is larger than 256x256 (your model’s input size), split it into tiles.
- Use a sliding window or grid approach.

```python
from itertools import product

def tile_image(image, tile_size=256):
    h, w, c = image.shape
    tiles = []
    for i, j in product(range(0, h, tile_size), range(0, w, tile_size)):
        tile = image[i:i+tile_size, j:j+tile_size, :]
        if tile.shape[:2] == (tile_size, tile_size):  # Ensure full tile
            tiles.append(tile)
    return tiles

pre_tiles = tile_image(pre_image)
post_tiles = tile_image(post_image)
print(f"Number of tiles: {len(pre_tiles)}")
```

#### 2.4 Prepare Input Pairs
- Your U-Net likely takes paired pre- and post-images stacked together (e.g., 6 channels: RGB_pre + RGB_post).
- Stack tiles accordingly.

```python
input_pairs = [np.concatenate([pre_tile, post_tile], axis=2) for pre_tile, post_tile in zip(pre_tiles, post_tiles)]
```

#### 2.5 Normalize for Model Input
- Apply ImageNet normalization (or whatever your model expects).

```python
from torchvision import transforms

normalize = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) and normalizes to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406] * 2,  # 6 channels
                         std=[0.229, 0.224, 0.225] * 2)
])

input_tensors = [normalize(pair.astype(np.float32)) for pair in input_pairs]
```

---

### Step 3: Load Your Pre-Trained Model
- Load the model weights trained on OSCD.

```python
import torch
from segmentation_models_pytorch import Unet  # Assuming you used this library

# Load model (adjust encoder_name if different)
model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=1, activation=None)
model.load_state_dict(torch.load("path/to/your_model_weights.pth", map_location="cpu"))
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

### Step 4: Run Inference on Your Custom Dataset
- Predict change masks for each tile.

```python
predictions = []
with torch.no_grad():
    for input_tensor in input_tensors:
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_tensor)  # Shape: (1, 1, H, W)
        pred_mask = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid for probability
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold to binary mask
        predictions.append(pred_mask[0, 0])  # Remove batch and channel dims
```

---

### Step 5: Reconstruct and Visualize Results
- Stitch tiles back into the full AOI and visualize changes.

#### 5.1 Reconstruct Full Mask
```python
def reconstruct_tiles(tiles, original_shape, tile_size=256):
    h, w = original_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i, j in product(range(0, h, tile_size), range(0, w, tile_size)):
        if idx < len(tiles):
            full_mask[i:i+tile_size, j:j+tile_size] = tiles[idx]
            idx += 1
    return full_mask

change_mask = reconstruct_tiles(predictions, pre_image.shape)
```

#### 5.2 Visualize
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Pre-Event (2023)")
plt.imshow(pre_image)
plt.subplot(1, 3, 2)
plt.title("Post-Event (2025)")
plt.imshow(post_image)
plt.subplot(1, 3, 3)
plt.title("Predicted Change Mask")
plt.imshow(change_mask, cmap="gray")
plt.show()
```

#### 5.3 Save as GeoTIFF (Optional)
- Retain geospatial metadata for GIS analysis.

```python
with rasterio.open(pre_image_path) as src:
    meta = src.meta.copy()
    meta.update(count=1, dtype="uint8")  # Single-band mask

with rasterio.open("change_mask.tif", "w", **meta) as dst:
    dst.write(change_mask, 1)
```

---

### Step 6: Evaluate and Refine
- **Manual Validation:** Since you don’t have ground truth for your custom dataset yet, manually inspect the predicted mask against visible changes in RGB images.
- **Potential Issues:**
  - **Domain Shift:** OSCD (13 cities, curated) differs from your Sentinel-2 AOI (resolution, scene content). Your model may underperform due to this.
  - **Resolution:** Sentinel-2’s 10m resolution vs. OSCD’s mix of resolutions might affect feature extraction.
- **Next Steps:**
  - If results are poor, annotate a small subset of your AOI for ground truth (e.g., using QGIS or Labelme) and fine-tune the model.
  - Adjust preprocessing (e.g., resolution, normalization) to better match OSCD.

---

### Step 7: Hardware Considerations (GPU)
- **Inference:** For testing on a few tiles, even an RTX 4090 (24 GB VRAM) is overkill—any modern GPU (e.g., RTX 3060) or CPU will suffice.
- **Fine-Tuning (if needed):** If you annotate and retrain, the RTX 5090 or A100 would be ideal for faster training on larger batches. Your current dataset size seems small, so an RTX 4090 should still handle it efficiently.

---

### Summary of Next Steps
1. Preprocess Sentinel-2 GeoTIFFs (load, scale, tile, normalize).
2. Load your pre-trained U-Net model.
3. Run inference on tiled input pairs.
4. Reconstruct and visualize the change mask.
5. Validate manually and refine if needed (e.g., fine-tuning).

Let me know if you hit any roadblocks or need code tweaks for your specific setup! How large is your AOI, and do you plan to annotate ground truth?
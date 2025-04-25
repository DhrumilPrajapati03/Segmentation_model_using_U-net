I’ll simplify the code by removing error handling, debug output, and extra features like the TCI option, focusing only on the core functionality. Below are two concise scripts: one to plot R, G, B bands individually and another to combine them into an RGB composite. I’ll assume your files are in `/home/desktop/implementation/datasets/sentinel-2/ahmedabad/dhrumil` with names like `S2_20230222_B04.tif` (adjust as needed).

---

### Simplified Script 1: Plot R, G, B Bands Individually

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt

base_path = "/home/desktop/implementation/datasets/sentinel-2/ahmedabad/dhrumil"

# Load pre-event bands
pre_red = rasterio.open(f"{base_path}/S2_20230222_B04.tif").read(1)    # Red
pre_green = rasterio.open(f"{base_path}/S2_20230222_B03.tif").read(1)  # Green
pre_blue = rasterio.open(f"{base_path}/S2_20230222_B02.tif").read(1)   # Blue

# Load post-event bands
post_red = rasterio.open(f"{base_path}/S2_20250212_B04.tif").read(1)   # Red
post_green = rasterio.open(f"{base_path}/S2_20250212_B03.tif").read(1) # Green
post_blue = rasterio.open(f"{base_path}/S2_20250212_B02.tif").read(1)  # Blue

# Normalize to 0–255
max_val = 4000  # Adjust if needed
pre_red = (pre_red / max_val * 255).astype(np.uint8)
pre_green = (pre_green / max_val * 255).astype(np.uint8)
pre_blue = (pre_blue / max_val * 255).astype(np.uint8)
post_red = (post_red / max_val * 255).astype(np.uint8)
post_green = (post_green / max_val * 255).astype(np.uint8)
post_blue = (post_blue / max_val * 255).astype(np.uint8)

# Plot
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(pre_red, cmap="gray"); ax[0, 0].set_title("Pre Red (B4)"); ax[0, 0].axis("off")
ax[0, 1].imshow(pre_green, cmap="gray"); ax[0, 1].set_title("Pre Green (B3)"); ax[0, 1].axis("off")
ax[0, 2].imshow(pre_blue, cmap="gray"); ax[0, 2].set_title("Pre Blue (B2)"); ax[0, 2].axis("off")
ax[1, 0].imshow(post_red, cmap="gray"); ax[1, 0].set_title("Post Red (B4)"); ax[1, 0].axis("off")
ax[1, 1].imshow(post_green, cmap="gray"); ax[1, 1].set_title("Post Green (B3)"); ax[1, 1].axis("off")
ax[1, 2].imshow(post_blue, cmap="gray"); ax[1, 2].set_title("Post Blue (B2)"); ax[1, 2].axis("off")
plt.tight_layout()
plt.show()
```

---

### Simplified Script 2: Combine and Plot RGB Composite

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt

base_path = "/home/desktop/implementation/datasets/sentinel-2/ahmedabad/dhrumil"

# Load and combine pre-event bands
pre_red = rasterio.open(f"{base_path}/S2_20230222_B04.tif").read(1)
pre_green = rasterio.open(f"{base_path}/S2_20230222_B03.tif").read(1)
pre_blue = rasterio.open(f"{base_path}/S2_20230222_B02.tif").read(1)
pre_rgb = np.dstack((pre_red, pre_green, pre_blue))

# Load and combine post-event bands
post_red = rasterio.open(f"{base_path}/S2_20250212_B04.tif").read(1)
post_green = rasterio.open(f"{base_path}/S2_20250212_B03.tif").read(1)
post_blue = rasterio.open(f"{base_path}/S2_20250212_B02.tif").read(1)
post_rgb = np.dstack((post_red, post_green, post_blue))

# Normalize to 0–255
max_val = 4000  # Adjust if needed
pre_rgb = (pre_rgb / max_val * 255).astype(np.uint8)
post_rgb = (post_rgb / max_val * 255).astype(np.uint8)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(pre_rgb); ax[0].set_title("Pre RGB (22/02/2023)"); ax[0].axis("off")
ax[1].imshow(post_rgb); ax[1].set_title("Post RGB (12/02/2025)"); ax[1].axis("off")
plt.tight_layout()
plt.show()
```

---

### Key Points
- **Conciseness:** Reduced to minimal lines by skipping error checks, dynamic scaling, and extra options.
- **Normalization:** Hardcoded `max_val = 4000` (Level-2A default). Change to `10000` if using Level-1C data.
- **Filenames:** Update `S2_20230222_B04.tif`, etc., to match your actual files.
- **Dependencies:** Requires `rasterio`, `numpy`, and `matplotlib`.

These scripts should work if your files exist and match the expected format. If you still get black images or errors, share your exact filenames or the issue, and I’ll tweak further!


OSCD/
├── images/
│   ├── abidjan/
│   │   ├── imgs_1/
│   │   ├── imgs_2/
│   │   ├── imgs_1_rect/
│   │   ├── imgs_2_rect/
│   │   └── pair/
├── train_labels/
│   ├── abidjan/
│   │   └── cm/
│   │       └── change_mask.png
├── test_labels/
│   ├── bangkok/
│   │   └── cm/
│   │       └── change_mask.png
├── all.txt
├── train.txt
└── test.txt


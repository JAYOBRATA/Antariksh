import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.transform import resize

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_shape=None):
    with rasterio.open(image_path) as src:
        image = src.read(1)
        if target_shape:
            image = resize(image, target_shape, anti_aliasing=True, mode='reflect')
        return image

# Paths to the images
chandrayan_image_path = "C:\\Users\JAYOBRATA ROY\Downloads\ch2_tmc_ncn_20200208T0057596133_b_brw_m65.png"
lro_mosaic_path = "C:\\Users\JAYOBRATA ROY\Downloads\lro_wac_ref_img.jpg"

# Load images
chandrayan_image = load_and_preprocess_image(chandrayan_image_path)
lro_mosaic = load_and_preprocess_image(lro_mosaic_path)

# Resample Chandrayaan-2 image to match LRO mosaic resolution
chandrayan_resampled = resize(chandrayan_image, lro_mosaic.shape, anti_aliasing=True, mode='reflect')

# Normalize images using histogram equalization
chandrayan_equalized = cv2.equalizeHist((chandrayan_resampled * 255).astype(np.uint8))
lro_equalized = cv2.equalizeHist((lro_mosaic * 255).astype(np.uint8))

# Display the original and equalized images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(chandrayan_image, cmap='gray')
plt.title('Original Chandrayaan-2 Image')

plt.subplot(2, 2, 2)
plt.imshow(chandrayan_equalized, cmap='gray')
plt.title('Equalized Chandrayaan-2 Image')

plt.subplot(2, 2, 3)
plt.imshow(lro_mosaic, cmap='gray')
plt.title('Original LRO Mosaic')

plt.subplot(2, 2, 4)
plt.imshow(lro_equalized, cmap='gray')
plt.title('Equalized LRO Mosaic')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

def load_image(image_path):
    """Load an image from a file."""
    image = io.imread(image_path, as_gray=True)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image

def normalize_image(image):
    """Normalize the image to have values in the range [0, 255]."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)

def resample_image(image, target_shape):
    """Resize the image to match the target shape."""
    return resize(image, target_shape, anti_aliasing=True, mode='reflect')

def template_matching(tmc_image, lro_mosaic):
    """Perform template matching to find the best match location."""
    if tmc_image is None or lro_mosaic is None:
        raise ValueError("One or both images are None.")
    if tmc_image.shape[0] > lro_mosaic.shape[0] or tmc_image.shape[1] > lro_mosaic.shape[1]:
        raise ValueError("Template image is larger than the mosaic image.")
    
    # Perform template matching
    result = cv2.matchTemplate(lro_mosaic, tmc_image, cv2.TM_CCOEFF_NORMED)
    if result is None:
        raise ValueError("Template matching failed.")
        
    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, result

def plot_results(lro_mosaic, tmc_image, max_loc):
    """Plot the results of template matching."""
    h, w = tmc_image.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(lro_mosaic, cmap='gray')
    plt.title('LRO Mosaic')
    plt.gca().add_patch(plt.Rectangle(top_left, w, h, edgecolor='r', facecolor='none'))

    plt.subplot(1, 2, 2)
    plt.imshow(tmc_image, cmap='gray')
    plt.title('Chandrayaan-2 TMC Image')
    
    plt.show()

# Paths to your images
tmc_image_path = "C:\\Users\JAYOBRATA ROY\Downloads\ch2_tmc_ncn_20200208T0057596133_b_brw_m65.png"
lro_mosaic_path = "C:\\Users\JAYOBRATA ROY\Downloads\lro_wac_ref_img.jpg"

# Load images
tmc_image = load_image(tmc_image_path)
lro_mosaic = load_image(lro_mosaic_path)

# Normalize images
tmc_image_normalized = normalize_image(tmc_image)
lro_mosaic_normalized = normalize_image(lro_mosaic)

# Resample TMC image to LRO mosaic resolution
tmc_image_resampled = resample_image(tmc_image_normalized, lro_mosaic_normalized.shape)

# Perform template matching
max_loc, result = template_matching(tmc_image_resampled, lro_mosaic_normalized)

# Plot the results
plot_results(lro_mosaic_normalized, tmc_image_resampled, max_loc)

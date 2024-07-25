import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images using OpenCV
tmc_image_path = "C:\\Users\JAYOBRATA ROY\Downloads\ch2_tmc_ncn_20200208T0057596133_d_img_m65\browse\calibrated\20200208\ch2_tmc_ncn_20200208T0057596133_b_brw_m65.png"
lro_mosaic_path = "C:\\Users\JAYOBRATA ROY\Downloads\lro_wac_ref_img.jpg"

tmc_image = cv2.imread(tmc_image_path, cv2.IMREAD_GRAYSCALE)
lro_mosaic = cv2.imread(lro_mosaic_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if tmc_image is None:
    print(f"Failed to load TMC image: {tmc_image_path}")
if lro_mosaic is None:
    print(f"Failed to load LRO mosaic: {lro_mosaic_path}")

# Apply histogram equalization
equalized_tmc_image = cv2.equalizeHist(tmc_image)
equalized_lro_mosaic = cv2.equalizeHist(lro_mosaic)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(tmc_image, cmap='gray')
plt.title('Original TMC Image')

plt.subplot(2, 2, 2)
plt.imshow(equalized_tmc_image, cmap='gray')
plt.title('Equalized TMC Image')

plt.subplot(2, 2, 3)
plt.imshow(lro_mosaic, cmap='gray')
plt.title('Original LRO Mosaic')

plt.subplot(2, 2, 4)
plt.imshow(equalized_lro_mosaic, cmap='gray')
plt.title('Equalized LRO Mosaic')

plt.tight_layout()
plt.show()

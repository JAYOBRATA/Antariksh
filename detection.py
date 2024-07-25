import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to your images
tmc_image_path = "C:\\Users\JAYOBRATA ROY\Downloads\ch2_tmc_ncn_20200208T0057596133_b_brw_m65.png"
lro_mosaic_path = "C:\\Users\JAYOBRATA ROY\Downloads\lro_wac_ref_img.jpg"

# Load images in grayscale
tmc_image = cv2.imread(tmc_image_path, cv2.IMREAD_GRAYSCALE)
lro_mosaic = cv2.imread(lro_mosaic_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if tmc_image is None:
    print(f"Failed to load TMC image: {tmc_image_path}")
if lro_mosaic is None:
    print(f"Failed to load LRO mosaic: {lro_mosaic_path}")

# Create ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(tmc_image, None)
kp2, des2 = orb.detectAndCompute(lro_mosaic, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
matched_image = cv2.drawMatches(tmc_image, kp1, lro_mosaic, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched features
plt.figure(figsize=(15, 10))
plt.imshow(matched_image)
plt.title('Matched Features between TMC and LRO Mosaic')
plt.show()

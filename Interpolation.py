import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\JAYOBRATA ROY\Downloads\img.jpg"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Could not open or find the image at {image_path}")
else:
    # Define the new size for resizing
    new_size = (image.shape[1] // 20, image.shape[0] // 20)

    # Bilinear interpolation
    resampled_bilinear = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Bicubic interpolation
    resampled_bicubic = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    # Display the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(resampled_bilinear, cv2.COLOR_BGR2RGB))
    plt.title('Bilinear Interpolation')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resampled_bicubic, cv2.COLOR_BGR2RGB))
    plt.title('Bicubic Interpolation')
    

    plt.show()



# The x-axis represents the horizontal pixel positions in the image. The values start from 0 on the left and increase to the right, up to the width of the image in pixels.

# The y-axis represents the vertical pixel positions in the image. The values start from 0 at the top and increase downwards, up to the height of the image in pixels.
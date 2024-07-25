import cv2

# Load the main image and the template image
main_image = cv2.imread("C:\\Users\JAYOBRATA ROY\Downloads\ch2_tmc_ncn_20200208T0057596133_b_brw_m65.png", 0)
template = cv2.imread("C:\\Users\JAYOBRATA ROY\Downloads\lro_wac_ref_img.jpg", 0)

# Check the dimensions
main_height, main_width = main_image.shape
templ_height, templ_width = template.shape

# Resize the template if it is larger than the main image
if templ_height > main_height or templ_width > main_width:
    scale_percent = 50  # percentage of the original size
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

# Now apply template matching
res = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

# Draw rectangles around matched areas
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 0, 255), 2)

# Show the result
plt.imshow(main_image, cmap='gray')
plt.title('Detected Craters')
plt.show()

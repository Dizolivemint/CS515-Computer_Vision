import cv2
import numpy as np

# Load the image
image_path = 'mycats.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Image not found at {image_path}")

# Apply binary thresholding to convert to a binary image
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Define the structuring element
kernel = np.ones((3, 3), np.uint8)

# Perform morphological transformations
dilated = cv2.dilate(binary_image, kernel, iterations=2)  # Dilation to enlarge shapes
eroded = cv2.erode(binary_image, kernel, iterations=2)    # Erosion to shrink shapes

# Create a morphological gradient to extract contours
contours = cv2.absdiff(dilated, eroded)

# Invert the colors to get a stylized outline effect
stylized_image = 255 - contours

# Display the result
cv2.imshow('Stylized Outline Effect', stylized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output
cv2.imwrite('stylized_outline.jpg', stylized_image)

import cv2
import numpy as np

# Load the image
image_path = 'mycats.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Image not found at {image_path}")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the structuring element (kernel)
kernel = np.ones((3, 3), np.uint8)

# Apply dilation and erosion
dilated = cv2.dilate(gray_image, kernel, iterations=1)
eroded = cv2.erode(gray_image, kernel, iterations=1)

# Create a Morphological Gradient to emphasize edges
morph_gradient = cv2.subtract(dilated, eroded)

# Enhance the contrast by combining original and morph gradient
# Weights can be adjusted for different intensity levels
pseudo_hdr = cv2.addWeighted(gray_image, 0.7, morph_gradient, 0.3, 0)

# Convert the enhanced image back to BGR for visualization
pseudo_hdr_color = cv2.cvtColor(pseudo_hdr, cv2.COLOR_GRAY2BGR)

# Stack the original and pseudo-HDR images side-by-side for comparison
combined = np.hstack((image, pseudo_hdr_color))

# Display the result
cv2.imshow('Original vs Pseudo-HDR', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('pseudo_hdr_output.jpg', pseudo_hdr_color)

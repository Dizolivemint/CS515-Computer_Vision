import cv2
import numpy as np

def isolate_cats(image_path, threshold, kernel_size):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    # Apply erosion
    mask = cv2.erode(dilation, kernel, iterations=1)
    
    # Create a 4-channel image (BGRA) for the result
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Apply the mask to the alpha channel
    result[:, :, 3] = mask
    
    return result, mask

# Parameters (easily adjustable)
image_path = 'mycats.jpg'
threshold = 100  # Adjust this value to refine the threshold
kernel_size = 5  # Adjust this value to change the size of the kernel

# Process the image
result, mask = isolate_cats(image_path, threshold, kernel_size)S

# Save results
cv2.imwrite('mask.png', mask)
cv2.imwrite('isolated_cats.png', result)
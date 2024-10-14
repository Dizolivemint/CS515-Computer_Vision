import cv2
import numpy as np
import os
import glob

def load_source_image():
  # Define directories for source and target images with backslashes
  source_directory = os.path.join('.', 'source')

  # Get the first image in the source directory (either .jpg or .png)
  source_image_list = glob.glob(os.path.join(source_directory, '*.[jp][pn]g'))  # Matches .jpg and .png
  if len(source_image_list) == 0:
      raise Exception(f"No images found in source directory: {os.path.abspath(source_directory)}")
  source_image_path = source_image_list[0]

  # Replace slashes with backslashes for Windows compatibility
  source_image_path = source_image_path.replace('/', '\\')

  # Debugging prints for loaded file paths
  print(f"Using source image: {os.path.abspath(source_image_path)}")

  # Check if images are correctly found
  if not os.path.isfile(source_image_path):
      raise FileNotFoundError(f"Source image not found at: {source_image_path}")

  return cv2.imread(source_image_path)

def ocr(image, threshold, kernel_size):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
   
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
   
    # Apply dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    # Show the dilation result
    cv2.imshow('Dilation', dilation)
   
    # Apply erosion
    erosion = cv2.erode(binary, kernel, iterations=1)
    
    # Show the erosion result
    cv2.imshow('Erosion', erosion)
    
    # Apply opening
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Show the opening result
    cv2.imshow('Opening', opening)
    
    # Apply closing
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Show the closing result
    cv2.imshow('Closing', closing)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    return

# Parameters (easily adjustable)
image = load_source_image()
threshold = 100  # Adjust this value to refine the threshold
kernel_size = 5  # Adjust this value to change the size of the kernel

# Process the image
ocr(image, threshold, kernel_size)
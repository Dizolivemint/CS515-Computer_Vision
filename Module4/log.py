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

def apply_filters(image, kernel_size, sigma=1):
    # Gaussian filter
    gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)

    # Gaussian followed by Laplacian
    gaussian_laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=kernel_size)

    return gaussian, laplacian, gaussian_laplacian

def display_images(images, titles):
    for i, (img, title) in enumerate(zip(images, titles)):
        cv2.imshow(title, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def __init__():
  # Load the source image
  source_image = load_source_image()

  # Display the source image
  cv2.imshow('Source Image', source_image)
  
  # Convert the image to grayscale
  source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
  
  # Set sigma value
  sigma = 1
  
  # 3x3 kernel
  kernel_size = 3
  gaussian, laplacian, gaussian_laplacian = apply_filters(source_image, kernel_size, sigma)
  display_images([gaussian, laplacian, gaussian_laplacian], ['Gaussian Filter', 'Laplacian Filter', 'Gaussian-Laplacian Filter'])
  
  # 5x5 kernel
  kernel_size = 5
  gaussian, laplacian, gaussian_laplacian = apply_filters(source_image, kernel_size, sigma)
  display_images([gaussian, laplacian, gaussian_laplacian], ['Gaussian Filter', 'Laplacian Filter', 'Gaussian-Laplacian Filter'])
  
  # 7x7 kernel
  kernel_size = 7
  gaussian, laplacian, gaussian_laplacian = apply_filters(source_image, kernel_size, sigma)
  display_images([gaussian, laplacian, gaussian_laplacian], ['Gaussian Filter', 'Laplacian Filter', 'Gaussian-Laplacian Filter'])
  
__init__()
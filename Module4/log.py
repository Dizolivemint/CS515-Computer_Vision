import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def display_images(image, sigma):
    # Create figure for subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Image Filtering Comparison with Sigma={sigma}', fontsize=16)

    # Apply filters for each kernel size
    kernels = [3, 5, 7]
    filter_names = ["Gaussian", "Laplacian", "Gaussian + Laplacian"]

    for i, kernel_size in enumerate(kernels):
        gaussian, laplacian, gaussian_laplacian = apply_filters(image, kernel_size, sigma)

        # Place images in subplots
        axs[i, 0].imshow(gaussian, cmap='gray')
        axs[i, 0].set_title(f'Gaussian {kernel_size}x{kernel_size}')
        axs[i, 1].imshow(laplacian, cmap='gray')
        axs[i, 1].set_title(f'Laplacian {kernel_size}x{kernel_size}')
        axs[i, 2].imshow(gaussian_laplacian, cmap='gray')
        axs[i, 2].set_title(f'Gauss + Laplacian {kernel_size}x{kernel_size}')

        # Set row labels
        axs[i, 0].set_ylabel(f'{kernel_size}x{kernel_size} Kernel', fontsize=12)

    # Set column labels
    for j, col in enumerate(filter_names):
        axs[0, j].set_xlabel(col, fontsize=14)

    # Remove axis
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    
def __init__():
  # Load the source image
  source_image = load_source_image()

  # Display the source image
  cv2.imshow('Source Image', source_image)
  
  # Convert the image to grayscale
  source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
  
  display_images(source_image, sigma=1)
  display_images(source_image, sigma=2)
  display_images(source_image, sigma=3)
  display_images(source_image, sigma=4)
  display_images(source_image, sigma=5)
  
__init__()
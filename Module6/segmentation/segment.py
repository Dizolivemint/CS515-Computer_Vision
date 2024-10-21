import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

def get_sources(directory='.'):
    """
    Get the source images and the source directory.
    
    Returns: source_directory and source_image_list
    """
    source_directory = os.path.abspath(os.path.join(directory, 'source'))
    source_image_list = glob.glob(os.path.join(source_directory, '*.[jp][pn]g'))
    
    if not source_image_list:
        raise FileNotFoundError(f"No images found in source directory: {source_directory}")
    
    return source_directory, source_image_list

def load_source_image(source_directory, source_image_list, index=0):
    """
    Load the image from the specified directory.
    
    Returns:
    numpy.ndarray: The loaded image as a numpy array.
    """
    
    if not source_image_list:
        raise FileNotFoundError(f"No images found in source directory: {source_directory}")
    
    source_image_path = source_image_list[index]
    print(f"Using source image: {source_image_path}")
    
    return cv2.imread(source_image_path)

def preprocess_image(image):
    """
    Preprocess the image for better segmentation.
    
    Args:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The preprocessed grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def adaptive_threshold(image, method='gaussian', block_size=11, c=2):
    """
    Apply adaptive thresholding to the image.
    
    Args:
    image (numpy.ndarray): The preprocessed grayscale image.
    method (str): The thresholding method ('mean' or 'gaussian').
    block_size (int): Size of a pixel neighborhood that is used to calculate a threshold value.
    c (int): Constant subtracted from the mean or weighted mean.
    
    Returns:
    numpy.ndarray: The thresholded binary image.
    """
    if method == 'mean':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, block_size, c)
    elif method == 'gaussian':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c)
    else:
        raise ValueError("Invalid method. Choose 'mean' or 'gaussian'.")

def otsu_threshold(image):
    """
    Apply Otsu's thresholding to the image.
    
    Args:
    image (numpy.ndarray): The preprocessed grayscale image.
    
    Returns:
    numpy.ndarray: The thresholded binary image.
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def local_otsu_threshold(image, tile_size=(8, 8)):
    """
    Apply local Otsu's thresholding to the image.
    
    Args:
    image (numpy.ndarray): The preprocessed grayscale image.
    tile_size (tuple): Size of the local tiles for thresholding.
    
    Returns:
    numpy.ndarray: The thresholded binary image.
    """
    rows, cols = image.shape
    tile_rows, tile_cols = tile_size
    
    result = np.zeros_like(image)
    
    for i in range(0, rows, tile_rows):
        for j in range(0, cols, tile_cols):
            tile = image[i:i+tile_rows, j:j+tile_cols]
            _, local_thresh = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result[i:i+tile_rows, j:j+tile_cols] = local_thresh
    
    return result

def display_results(results):
    """
    Display the results of various image processing operations.
    
    Args:
    results (dict): A dictionary containing the processed images.
    """
    num_images = len(results)
    cols = 2
    rows = (num_images + 1) // 2
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (name, image) in enumerate(results.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    source_directory, source_image_list = get_sources()
    
    for i in range(len(source_image_list)):
        print(f"Processing image {i + 1}...")
        
        # Load the source image
        original_image = load_source_image(source_directory, source_image_list, i)
        
        # Preprocess the image
        preprocessed = preprocess_image(original_image)
        
        # Apply different thresholding techniques
        results = {
            'Original': cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            'Preprocessed': preprocessed,
            'Adaptive (Mean)': adaptive_threshold(preprocessed, method='mean'),
            'Adaptive (Gaussian)': adaptive_threshold(preprocessed, method='gaussian'),
            'Otsu': otsu_threshold(preprocessed),
            'Local Otsu': local_otsu_threshold(preprocessed)
        }
        
        # Display results
        display_results(results)

if __name__ == "__main__":
    main()
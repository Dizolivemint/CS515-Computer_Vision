import cv2
import numpy as np
import os
import glob

def load_source_image(directory='.'):
    """
    Load the first image from the specified directory.
    
    Args:
    directory (str): The directory to search for images. Defaults to current directory.
    
    Returns:
    numpy.ndarray: The loaded image as a numpy array.
    """
    source_directory = os.path.abspath(os.path.join(directory, 'source'))
    source_image_list = glob.glob(os.path.join(source_directory, '*.[jp][pn]g'))
    
    if not source_image_list:
        raise FileNotFoundError(f"No images found in source directory: {source_directory}")
    
    source_image_path = source_image_list[0]
    print(f"Using source image: {source_image_path}")
    
    return cv2.imread(source_image_path)

def preprocess_image(image):
    """
    Preprocess the image for better cursive text extraction.
    
    Args:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The preprocessed grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def thinning(image):
    """
    Perform thinning operation on the image.
    
    Args:
    image (numpy.ndarray): The input binary image.
    
    Returns:
    numpy.ndarray: The thinned image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    thin = np.zeros(image.shape, dtype='uint8')
    while True:
        eroded = cv2.erode(image, kernel)
        opened = cv2.dilate(eroded, kernel)
        subset = image - opened
        thin = cv2.bitwise_or(thin, subset)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return thin

def fill_holes(image):
    """
    Fill holes in the text using morphological operations.
    
    Args:
    image (numpy.ndarray): The input binary image.
    
    Returns:
    numpy.ndarray: The image with filled holes.
    """
    # Perform morphological closing
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Fill holes
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(closing.shape, np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, (255), -1)
    
    return mask

def enhance_cursive(image, kernel_size=3):
    """
    Enhance cursive text using morphological operations.
    
    Args:
    image (numpy.ndarray): The preprocessed binary image.
    kernel_size (int): Size of the kernel for morphological operations.
    
    Returns:
    dict: A dictionary containing the results of various morphological operations.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    thinned = thinning(image)
    filled = fill_holes(erosion)  # Apply hole filling to the eroded image
    
    return {
        'original': image,
        'dilation': dilation,
        'erosion': erosion,
        'opening': opening,
        'closing': closing,
        'thinned': thinned,
        'filled_erosion': filled  # Add the new filled erosion result
    }

def display_results(results):
    """
    Display the results of various image processing operations.
    
    Args:
    results (dict): A dictionary containing the processed images.
    """
    for name, image in results.items():
        cv2.imshow(name.capitalize(), image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the source image
    original_image = load_source_image()
    
    # Preprocess the image
    preprocessed = preprocess_image(original_image)
    
    # Enhance cursive text
    cursive_enhancements = enhance_cursive(preprocessed)
    
    # Add original color image to results
    cursive_enhancements['original_color'] = original_image
    
    # Display results
    display_results(cursive_enhancements)

if __name__ == "__main__":
    main()
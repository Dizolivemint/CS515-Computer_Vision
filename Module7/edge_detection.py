import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def create_synthetic_image(width=200, height=200, background_intensity=128):
    """
    Generate a synthetic image with a square and circle.
    """
    # Create grayscale image
    image = np.full((height, width), background_intensity, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (100, 100), color=255, thickness=-1)
    cv2.circle(image, (150, 150), radius=25, color=255, thickness=-1)
    return image

def create_color_intensity_image(width=200, height=200):
    """
    Create an image with different colors for background, square, and circle.
    Returns both the colored image and its grayscale version for edge detection.
    """
    # Create color image (BGR format)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Light blue background (BGR: 230, 216, 173)
    image[:, :] = [230, 216, 173]
    
    # Dark blue square (BGR: 139, 0, 0)
    cv2.rectangle(image, (50, 50), (100, 100), color=(139, 0, 0), thickness=-1)
    
    # Dark red circle (BGR: 0, 0, 139)
    cv2.circle(image, (150, 150), radius=25, color=(0, 0, 139), thickness=-1)
    
    # Convert to grayscale for edge detection
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, grayscale

def apply_edge_detection(image, method='canny', threshold1=50, threshold2=150):
    """
    Apply different edge detection methods to the image.
    """
    if method.lower() == 'canny':
        return cv2.Canny(image, threshold1, threshold2)
    elif method.lower() == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges / edges.max() * 255)
        _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)
        return edges
    elif method.lower() == 'laplacian':
        edges = cv2.Laplacian(image, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        _, edges = cv2.threshold(edges, threshold1, 255, cv2.THRESH_BINARY)
        return edges
    else:
        raise ValueError("Unsupported edge detection method")

def calculate_f1_score(ground_truth, detected_edges):
    """
    Calculate F1 score between ground truth and detected edges.
    """
    gt_binary = (ground_truth > 0).astype(np.int8)
    det_binary = (detected_edges > 0).astype(np.int8)
    gt_flat = gt_binary.ravel()
    det_flat = det_binary.ravel()
    try:
        return f1_score(gt_flat, det_flat)
    except Exception as e:
        print(f"Error calculating F1 score: {e}")
        return 0.0

def calculate_pratt_fom(ground_truth, detected_edges, alpha=1/9):
    """
    Calculate Pratt's Figure of Merit.
    """
    gt_coords = np.argwhere(ground_truth > 0)
    det_coords = np.argwhere(detected_edges > 0)
    
    if len(det_coords) == 0:
        return 0.0
    
    max_dist = np.sqrt(ground_truth.shape[0]**2 + ground_truth.shape[1]**2)
    sum_distances = 0
    
    for det_point in det_coords:
        distances = np.sqrt(np.sum((gt_coords - det_point)**2, axis=1))
        min_distance = np.min(distances) if len(distances) > 0 else max_dist
        sum_distances += 1 / (1 + alpha * min_distance**2)
    
    return sum_distances / max(len(gt_coords), len(det_coords))

def add_noise(image, noise_level=10):
    """
    Add Gaussian noise to the image.
    """
    if len(image.shape) == 3:  # Color image
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:  # Grayscale image
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

def evaluate_edge_detection(image, ground_truth, method, thresholds):
    """
    Evaluate edge detection performance across different thresholds.
    """
    f1_scores = []
    pratt_scores = []
    
    for t in thresholds:
        edges = apply_edge_detection(image, method, t, t*3)
        f1 = calculate_f1_score(ground_truth, edges)
        pratt = calculate_pratt_fom(ground_truth, edges)
        f1_scores.append(f1)
        pratt_scores.append(pratt)
    
    return f1_scores, pratt_scores

def plot_results(thresholds, results, title):
    """
    Plot evaluation results.
    """
    plt.figure(figsize=(10, 6))
    for method, scores in results.items():
        plt.plot(thresholds, scores, label=method)
    plt.xlabel('Threshold Value')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_edges(image, edges, title):
    """
    Visualize original image and detected edges side by side.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title(f'{title} Edges')
    plt.axis('off')
    plt.show()

def main():
    # Create original image and ground truth
    original_image = create_synthetic_image()
    ground_truth = apply_edge_detection(original_image, 'canny', 10, 30)
    
    # Create color intensity image
    color_image, color_gray = create_color_intensity_image()
    
    # Create noisy images
    noisy_original = add_noise(original_image)
    noisy_color = add_noise(color_image)
    noisy_color_gray = cv2.cvtColor(noisy_color, cv2.COLOR_BGR2GRAY)
    
    # Define threshold range for evaluation
    thresholds = list(range(10, 200, 10))
    
    # Evaluate all methods on all image variants
    image_variants = {
        'Original': (original_image, original_image),
        'Noisy': (noisy_original, noisy_original),
        'Color Intensity': (color_image, color_gray)
    }
    
    methods = ['canny', 'sobel', 'laplacian']
    
    for variant_name, (display_img, process_img) in image_variants.items():
        print(f"\nEvaluating {variant_name} Image:")
        f1_results = {}
        pratt_results = {}
        
        for method in methods:
            f1_scores, pratt_scores = evaluate_edge_detection(process_img, ground_truth, method, thresholds)
            f1_results[method] = f1_scores
            pratt_results[method] = pratt_scores
            
            # Print best scores
            best_f1 = max(f1_scores)
            best_threshold = thresholds[np.argmax(f1_scores)]
            print(f"{method.capitalize()} - Best F1: {best_f1:.3f} at threshold {best_threshold}")
            
            # Visualize best result
            best_edges = apply_edge_detection(process_img, method, best_threshold, best_threshold*3)
            visualize_edges(display_img, best_edges, f'{method.capitalize()} ({variant_name})')
        
        # Plot results
        plot_results(thresholds, f1_results, f'F1 Scores - {variant_name} Image')
        plot_results(thresholds, pratt_results, f'Pratt FOM - {variant_name} Image')

if __name__ == "__main__":
    main()
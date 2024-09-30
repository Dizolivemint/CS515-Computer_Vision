import cv2
import numpy as np

def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    sh, sw = target_size
    
    # Calculate scaling factor
    scale = min(sh/h, sw/w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize the image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create a black canvas of the target size
    padded = np.zeros((sh, sw, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_h = (sh - new_h) // 2
    pad_w = (sw - new_w) // 2
    
    # Place the resized image on the canvas
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded

# Load the two images
source_image = cv2.imread('PXL_20240929_215502349.MP.png')
target_image = cv2.imread('A_man_standing_confidently_in_a_full-body_portrait.jpg')

# Resize and pad source image to match target image dimensions
source_image = resize_and_pad(source_image, target_image.shape[:2])

# Convert images to grayscale
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

# Use Haar cascades to detect faces in both images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the target image
target_faces = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5)

# Detect faces in the source image
source_faces = face_cascade.detectMultiScale(source_gray, scaleFactor=1.1, minNeighbors=5)

# Ensure faces are detected in both images
if len(target_faces) == 0 or len(source_faces) == 0:
    raise Exception("Could not detect faces in one of the images")

# Get the coordinates of the face in the target image
(target_x, target_y, target_w, target_h) = target_faces[0]

# Get the coordinates of the face in the source image
(source_x, source_y, source_w, source_h) = source_faces[0]

# Print dimensions for debugging
print(f"Target Face Coordinates: x={target_x}, y={target_y}, w={target_w}, h={target_h}")
print(f"Source Face Coordinates: x={source_x}, y={source_y}, w={source_w}, h={source_h}")
print(f"Target Image Shape: {target_image.shape}")
print(f"Source Image Shape: {source_image.shape}")

# Crop the face region from the source image
source_face = source_image[source_y:source_y+source_h, source_x:source_x+source_w]

# Resize the source face to match the dimensions of the target face
resized_source_face = cv2.resize(source_face, (target_w, target_h))

# Create a mask for the full target image
full_target_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

# Create an elliptical mask for better blending
center = (target_w // 2, target_h // 2)
axes = (target_w // 2, target_h // 2)
cv2.ellipse(full_target_mask, (target_x + center[0], target_y + center[1]), axes, 0, 0, 360, (255, 255, 255), -1)

# Match histograms to better blend the source face into the target image
source_face_histogram_equalized = cv2.equalizeHist(cv2.cvtColor(resized_source_face, cv2.COLOR_BGR2GRAY))
source_face_final = cv2.merge((source_face_histogram_equalized, source_face_histogram_equalized, source_face_histogram_equalized))

# Calculate the center of the target face region for seamless cloning
center = (target_x + target_w // 2, target_y + target_h // 2)
print(f"Calculated Center: {center}")

# Ensure source_face_final matches the dimensions of the target face region
source_face_final = cv2.resize(source_face_final, (target_w, target_h))

# Create a region of interest (ROI) in the target image
roi = target_image[target_y:target_y+target_h, target_x:target_x+target_w]

# Perform seamless cloning on the ROI
output_roi = cv2.seamlessClone(source_face_final, roi, full_target_mask[target_y:target_y+target_h, target_x:target_x+target_w], (target_w//2, target_h//2), cv2.NORMAL_CLONE)

# Place the output ROI back into the target image
output_image = target_image.copy()
output_image[target_y:target_y+target_h, target_x:target_x+target_w] = output_roi

# Display the result
cv2.imshow("Face Swapped Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final output
cv2.imwrite("face_swapped_output.jpg", output_image)
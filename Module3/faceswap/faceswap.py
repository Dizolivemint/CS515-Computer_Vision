import cv2
import numpy as np

# Load the two images
source_image = cv2.imread('PXL_20240929_215502349.MP.png')
target_image = cv2.imread('A_man_standing_confidently_in_a_full-body_portrait.jpg')

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

# Analyze the target image's face using Canny edge detection
target_face_region = target_gray[target_y:target_y+target_h, target_x:target_x+target_w]
target_face_edges = cv2.Canny(target_face_region, 100, 200)

# Detect contours of the target face for edge matching
contours, _ = cv2.findContours(target_face_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ensure that we have contours before proceeding
if len(contours) > 0:
    # Convert contour points to a 2D array of integers
    target_contour_points = np.array(contours[0], dtype=np.int32)
else:
    raise Exception("No contours found in the target face region")

# Create a mask for the full target image (full size, not just the face region)
full_target_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

# Create a small mask for the target face region
target_face_mask = np.zeros_like(target_face_region)
cv2.fillConvexPoly(target_face_mask, target_contour_points, (255, 255, 255))

# Place the small face mask into the corresponding region of the full-size mask
full_target_mask[target_y:target_y+target_h, target_x:target_x+target_w] = target_face_mask

# Debugging dimensions and placements
print(f"Full Target Mask Shape: {full_target_mask.shape}")
print(f"Resized Source Face Shape: {resized_source_face.shape}")

# Optional: Match histograms to better blend the source face into the target image
source_face_histogram_equalized = cv2.equalizeHist(cv2.cvtColor(resized_source_face, cv2.COLOR_BGR2GRAY))
source_face_final = cv2.merge((source_face_histogram_equalized, source_face_histogram_equalized, source_face_histogram_equalized))

# Calculate the center of the target face region for seamless cloning
center_x = target_x + target_w // 2
center_y = target_y + target_h // 2

# Boundary check to ensure that the center point and source dimensions fit within the target image
image_height, image_width = target_image.shape[:2]

# Ensure the calculated center is within bounds
center_x = min(max(center_x, target_w // 2), image_width - target_w // 2)
center_y = min(max(center_y, target_h // 2), image_height - target_h // 2)

center = (center_x, center_y)

# Debugging center and dimensions
print(f"Calculated Center: {center}")
print(f"Target Image Width: {image_width}, Height: {image_height}")
print(f"Source Image Dimensions (w={target_w}, h={target_h})")

# Ensure the source face is in the same channel format as the target
if len(source_face_final.shape) == 2:
    source_face_final = cv2.cvtColor(source_face_final, cv2.COLOR_GRAY2BGR)

# Perform seamless cloning using the new full-size mask
output_image = cv2.seamlessClone(source_face_final, target_image, full_target_mask, center, cv2.NORMAL_CLONE)

# Display the result
cv2.imshow("Face Swapped Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final output
cv2.imwrite("face_swapped_output.jpg", output_image)

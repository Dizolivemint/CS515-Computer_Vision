import cv2
import numpy as np
import dlib
import os
import sys

def get_face_mask(img, landmarks):
    mask = np.zeros(img.shape[:2], dtype=np.float64)
    points = np.array(landmarks, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    return mask

def correct_colours(im1, im2, landmarks1):
    blur_amount = 0.6 * np.linalg.norm(
                              np.mean(landmarks1[36:42], axis=0) -
                              np.mean(landmarks1[42:48], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def download_shape_predictor():
    import urllib.request
    import bz2

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"
    dat_file = "shape_predictor_68_face_landmarks.dat"

    print("Downloading shape predictor file...")
    urllib.request.urlretrieve(url, bz2_file)

    print("Extracting shape predictor file...")
    with bz2.BZ2File(bz2_file) as fr, open(dat_file, 'wb') as fw:
        fw.write(fr.read())

    os.remove(bz2_file)
    print("Shape predictor file downloaded and extracted successfully.")

def align_eyes(source_image, source_landmarks, target_landmarks, target_shape):
    # Get eye landmarks
    left_eye_source = np.mean(source_landmarks[36:42], axis=0)
    right_eye_source = np.mean(source_landmarks[42:48], axis=0)
    left_eye_target = np.mean(target_landmarks[36:42], axis=0)
    right_eye_target = np.mean(target_landmarks[42:48], axis=0)

    # Calculate angle for rotation
    source_angle = np.degrees(np.arctan2(right_eye_source[1] - left_eye_source[1],
                                         right_eye_source[0] - left_eye_source[0]))
    target_angle = np.degrees(np.arctan2(right_eye_target[1] - left_eye_target[1],
                                         right_eye_target[0] - left_eye_target[0]))
    rotation_angle = target_angle - source_angle

    # Calculate scaling factor
    source_eye_distance = np.linalg.norm(right_eye_source - left_eye_source)
    target_eye_distance = np.linalg.norm(right_eye_target - left_eye_target)
    scale = target_eye_distance / source_eye_distance

    # Get the center of the source face
    source_center = np.mean(source_landmarks, axis=0)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(tuple(source_center), rotation_angle, scale)

    # Adjust the matrix to move the source face to the target face position
    target_center = np.mean(target_landmarks, axis=0)
    rotation_matrix[:, 2] += target_center - source_center

    # Apply the transformation
    aligned_image = cv2.warpAffine(source_image, rotation_matrix, (target_shape[1], target_shape[0]), borderMode=cv2.BORDER_REFLECT)

    return aligned_image
  
# Check if the shape predictor file exists, if not, download it
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    print(f"Shape predictor file not found: {predictor_path}")
    download_shape_predictor()

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError:
    print(f"Error loading the shape predictor. Please ensure {predictor_path} is in the current directory.")
    print("You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract the file and place it in the same directory as this script.")
    sys.exit(1)

# Load the two images
source_image_path = 'PXL_20240929_215502349.MP.png'
target_image_path = 'A_man_standing_confidently_in_a_full-body_portrait.jpg'

if not os.path.isfile(source_image_path) or not os.path.isfile(target_image_path):
    print(f"Error: Source image ({source_image_path}) or target image ({target_image_path}) not found.")
    print("Please ensure both images are in the current directory.")
    sys.exit(1)

source_image = cv2.imread(source_image_path)
target_image = cv2.imread(target_image_path)

# Detect faces in both images
source_faces = detector(source_image)
target_faces = detector(target_image)

if len(source_faces) == 0 or len(target_faces) == 0:
    print("Error: Could not detect faces in one or both of the images.")
    print("Please ensure the images contain clear, front-facing portraits.")
    sys.exit(1)

# Get the facial landmarks for both faces
source_landmarks = predictor(source_image, source_faces[0])
target_landmarks = predictor(target_image, target_faces[0])

# Convert landmarks to NumPy arrays
source_landmarks_points = np.array([(p.x, p.y) for p in source_landmarks.parts()])
target_landmarks_points = np.array([(p.x, p.y) for p in target_landmarks.parts()])

# Calculate affine transform for aligning eyes
left_eye_source = np.mean(source_landmarks_points[36:42], axis=0)
right_eye_source = np.mean(source_landmarks_points[42:48], axis=0)
left_eye_target = np.mean(target_landmarks_points[36:42], axis=0)
right_eye_target = np.mean(target_landmarks_points[42:48], axis=0)

source_eyes_center = (left_eye_source + right_eye_source) / 2
target_eyes_center = (left_eye_target + right_eye_target) / 2

angle = np.arctan2(right_eye_target[1] - left_eye_target[1],
                   right_eye_target[0] - left_eye_target[0]) - \
        np.arctan2(right_eye_source[1] - left_eye_source[1],
                   right_eye_source[0] - left_eye_source[0])

scale = np.linalg.norm(right_eye_target - left_eye_target) / \
        np.linalg.norm(right_eye_source - left_eye_source)

rotation_matrix = cv2.getRotationMatrix2D(tuple(source_eyes_center), np.degrees(angle), scale)
rotation_matrix[:, 2] += target_eyes_center - source_eyes_center

# Apply the affine transformation
aligned_source_image = cv2.warpAffine(source_image, rotation_matrix, (target_image.shape[1], target_image.shape[0]))

# Get the face mask for the aligned source image
aligned_source_face = detector(aligned_source_image)[0]
aligned_source_landmarks = predictor(aligned_source_image, aligned_source_face)
aligned_source_landmarks_points = np.array([(p.x, p.y) for p in aligned_source_landmarks.parts()])

mask = get_face_mask(aligned_source_image, aligned_source_landmarks_points)

# Correct colors
warped_corrected_img = correct_colours(target_image, aligned_source_image, target_landmarks_points)

# Combine the images
output = target_image * (1.0 - mask) + warped_corrected_img * mask

# Convert back to uint8
output = output.astype(np.uint8)

# Display the source
cv2.imshow("Source Image", source_image)
cv2.waitKey(0)

# Display the target
cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

# Display the result
cv2.imshow("Face Swapped Image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final output
cv2.imwrite("face_swapped_output.jpg", output)
print("Face swap completed successfully. Output saved as 'face_swapped_output.jpg'.")
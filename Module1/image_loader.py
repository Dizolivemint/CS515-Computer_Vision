import cv2
import os
import shutil

directory = './' # current directory
images_directory = './images/' # images directory
file_name = ''

# Scan the directory for the first .jpg file
for file in os.listdir(directory):
    if file.endswith('.jpg'):
        filename = file
        image_path = os.path.join(directory, filename)
        break

# Read the image
image = cv2.imread(image_path)

# Display the image
cv2.imshow('Image', image)

# Copy the image to the images directory
if not os.path.exists(images_directory):
    os.makedirs(images_directory)
shutil.copy(image_path, images_directory)

# Wait for a key press and close the window
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
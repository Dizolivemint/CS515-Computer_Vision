import cv2
import os
import numpy as np

def security_thread(image):
  # Apply edge detection
  edges = cv2.Canny(image, 100, 200)
  
  cv2.imshow('Security Thread Detection: Look for vertical thread', edges)
   
def watermark(image):
  # Apply thresholding to highlight the watermark
  _, thresh_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

  cv2.imshow('Watermark Detection: Look for watermark', thresh_image)
  
def color_shifting(image):
  # Apply histogram equalization to enhance color-shifting areas
  equalized_image = cv2.equalizeHist(image)
  
  cv2.imshow('Color Shifting Detection: Look for color-shifting areas', equalized_image)
  
def apply_transform(image):
  # Enlarge image
  image = cv2.resize(image, (0, 0), fx=4, fy=4)
  
  # Rotate the image by 90 degrees clockwise
  rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  
  return rotated_image

def main():
  # Set up directories
  current_directory = './'  # current directory

  # Find the first .jpg file in the current directory
  image_path = next(
      (os.path.join(current_directory, file) for file in os.listdir(current_directory) if file.endswith('.jpg')), 
      None
  )
  
  if image_path:
      # Read the image
      image = cv2.imread(image_path)
      
      color_image = apply_transform(image)
      gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
      
      security_thread(gray_image)
      watermark(gray_image)
      color_shifting(gray_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      
  else:
      print("No .jpg file found in the current directory.")
      
if __name__ == '__main__':
  main()
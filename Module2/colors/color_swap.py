import cv2
import os
import numpy as np


def extract_channels(image):
  # Split the image into its channels
  b, g, r = cv2.split(image)
    
  return r, g, b

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
      
      cv2.imshow('Original Image', image)
      
      # Extract the channels
      b, g, r = extract_channels(image)
      
      # Create blank channels (zero arrays) with the same shape as the original image
      zeros = np.zeros_like(r)

      # Create images where only one channel is active and the others are black
      red_channel = cv2.merge([zeros, zeros, r])
      green_channel = cv2.merge([zeros, g, zeros])
      blue_channel = cv2.merge([b, zeros, zeros])
      
      # Display the color channels
      cv2.imshow('Red Channel', red_channel)
      cv2.imshow('Green Channel', green_channel)
      cv2.imshow('Blue Channel', blue_channel)
      
      # Swap the red and green channels
      r, g = g, r
      
      new_image = cv2.merge([b, g, r])

      # Display the new image
      cv2.imshow('Image with Red and Green Channels Swapped', new_image)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      
  else:
      print("No .jpg file found in the current directory.")
      
if __name__ == '__main__':
  main()
### To use this script:

- Ensure you have the required libraries installed: cv2, numpy, and dlib.
- Install requirements.txt ```python -m pip install -r requirements.txt```
- Place your source image in source directory and target image in target directory
- The script will load the first file from the directory and output the file to the output directory
- Run the script. If the shape predictor file is missing, it will attempt to download it automatically.

### If you encounter any issues:

- Make sure your images are clear, front-facing portraits.
- Ensure you have an active internet connection for the automatic download of the shape predictor file.
- If the automatic download fails, you can manually download the file from the provided URL and place it in the script's directory.
import qrcode
import cv2
import numpy as np
from PIL import Image
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

# Step 1: Create a QR code with custom style
def generate_styled_qr_code(data, size=10):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=size,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create a styled QR code with rounded corners and radial gradient color mask
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask()
    )
    return img

# Step 2: Convert PIL Image to OpenCV format
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Step 3: Blend the QR code into the background image
def blend_images(qr_img, background_img_path, alpha=0.5):
    # Load background image
    background = cv2.imread(background_img_path)

    # Resize QR code to match the background size
    qr_img = cv2.resize(qr_img, (background.shape[1], background.shape[0]))

    # Blend the images
    blended = cv2.addWeighted(background, 1 - alpha, qr_img, alpha, 0)
    return blended

# Step 4: Save the resulting image
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# --- Main Script ---
# Data to encode in the QR code
data = "https://example.com"

# Generate styled QR code
qr_code_img = generate_styled_qr_code(data, size=15)

# Convert the QR code from PIL to OpenCV format
qr_code_cv2 = pil_to_cv2(qr_code_img)

# Blend the styled QR code into the background image
blended_image = blend_images(qr_code_cv2, "background.png", alpha=0.4)

# Save the final blended image
save_image(blended_image, "blended_styled_qr_code.png")

print("QR Code with custom styling blended successfully!")

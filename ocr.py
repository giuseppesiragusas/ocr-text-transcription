import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from pytesseract import Output

# Set the path to the tesseract executable
# You must set the path to where Tesseract is installed on your system for this to work
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    Perform preprocessing on the image to make it more suitable for OCR.
    - Convert to grayscale
    - Normalize the image to enhance contrast
    - Apply binary thresholding
    - Use Gaussian blur to reduce noise and improve text segmentation
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image
    norm_img = np.zeros((gray.shape[0], gray.shape[1]))
    gray = cv2.normalize(gray, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    # Apply thresholding to get a binary image
    _, thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Apply Gaussian blur to smooth the image
    blur_img = cv2.GaussianBlur(thresh_img, (1, 1), 0)
    
    return blur_img

def main():
    # Ask user for the image file path
    image_path = input("Please enter the path to the image file: ")
    
    # Strip quotes from the path in case they are included
    image_path = image_path.strip('"\'')

    # Preprocess the image to improve OCR accuracy
    preprocessed_image = preprocess_image(image_path)

    # Convert image to string using Tesseract OCR
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')
    
    # Print the transcribed text
    print("\nTranscribed text:")
    print(text)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()


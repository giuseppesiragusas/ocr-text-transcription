# The following program attempts to transcribe handwritten text in the most authentic way to the original, by using OCR Models.

# preliminary action: it is required to install pytesseract on local
# steps: 1) WIN+R 2) type 'cmd' and enter 3) type 'pip install pytesseract' and enter 4) run this code

#!/usr/bin/env python3

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    image = Image.open(image_path)
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply a threshold to get a binary image
    threshold = 200
    image = image.point(lambda p: p > threshold and 255)
    # Filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    return image

def main():
    image_path = input("Please enter the path to the image file: ")
    image_path = image_path.strip('"\'')

    # Preprocess the image
    image = preprocess_image(image_path)

    # OCR with Tesseract using a config that assumes a single column of text
    text = pytesseract.image_to_string(image, config='--psm 6')
    print("\nTranscribed text:")
    print(text)

if __name__ == "__main__":
    main()
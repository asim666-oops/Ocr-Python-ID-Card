from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

def preprocess_image(image):
    try:
        # Convert the image to a format OpenCV can use
        img = np.array(image)
        if img is None:
            raise ValueError("Image could not be loaded.")

        # Resize if too large
        max_dim = 3000
        if max(img.shape) > max_dim:
            scale = max_dim / max(img.shape)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Deskew image
        angle = get_skew_angle(enhanced)
        (h, w) = enhanced.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def get_skew_angle(image):
    # Convert to binary
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find all contours
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    angles = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Skip small contours
            continue
        # Fit a rotated rectangle
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    # Return the most common angle
    if angles:
        return np.median(angles)
    return 0

def extract_text(image):
    try:
        # Use PyTesseract to extract text from the image
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 3')
        return text
    except Exception as e:
        print(f"Error in extracting text: {e}")
        return ""

def clean_text(text):
    # Remove any non-alphanumeric characters except spaces, slashes, colons, and hyphens
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s/:-]', '', text)
    return cleaned_text

def structure_with_rules(extracted_text):
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Split the text into lines
    lines = cleaned_text.split('\n')

    # Initialize variables to store extracted information
    name = ""
    dob = ""
    gender = ""
    aadhaar_number = ""
    phone_number = ""

    # Define regular expression patterns for each field
    dob_pattern = r'DOB\s*:\s*(\d{2}\s*/\s*\d{2}\s*/\s*\d{4})'  # Adjusted to match DOB format with optional spaces around slashes
    gender_pattern = r'\b(Male|Female|MALE|FEMALE)\b'
    aadhaar_pattern = r'\b(\d{4}\s*\d{4}\s*\d{4})\b'
    phone_pattern = r'\b\d{10}\b'

    # Iterate over lines to extract information
    for i, line in enumerate(lines):
        if "Aadhaar no. issued" in line or "Issue Date" in line or "Download Date" in line:
            continue

        # Extract name based on position relative to DOB
        if not name and i + 1 < len(lines) and "DOB" in lines[i + 1]:
            name = line.strip()

        dob_match = re.search(dob_pattern, line)
        gender_match = re.search(gender_pattern, line)
        aadhaar_match = re.search(aadhaar_pattern, line)
        phone_match = re.search(phone_pattern, line)

        if dob_match:
            dob = dob_match.group(1).strip()  # Remove extra spaces
        if gender_match:
            gender = gender_match.group(0).strip()
        if aadhaar_match:
            aadhaar_number = aadhaar_match.group(1).replace(" ", "")
        if phone_match:
            phone_number = phone_match.group(0).strip()

    return {
        "name": name,
        "dob": dob,
        "gender": gender,
        "aadhaarNumber": aadhaar_number,
        "phoneNumber": phone_number
    }

@app.post("/extract/")
async def extract_info(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is not None:
            # Extract text from the image
            extracted_text = extract_text(preprocessed_image)
            # Structure the extracted information
            result = structure_with_rules(extracted_text)
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Image preprocessing failed"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# To run the application, use the command: uvicorn your_script_name:app --reload

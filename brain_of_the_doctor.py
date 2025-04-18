from dotenv import load_dotenv
load_dotenv()

import os
import sys
import base64
import time
import hashlib
from functools import lru_cache
from groq import Groq, GroqError

def encode_image(image_path, max_size=1024):
    """Convert image to base64 string with optional resizing"""
    try:
        import cv2
        # Read and optionally resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            
        # Encode directly from numpy array to base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception:
        # Fallback to original method if OpenCV fails
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

PRESCRIPTION_TEMPLATE = """
PRESCRIPTION
Date: {date}
Patient: {patient_name}
Diagnosis: {diagnosis}

Medications:
{medications}

Instructions:
{instructions}

Doctor: AI Doctor
"""

def generate_prescription(diagnosis, language="English"):
    """Generate a prescription based on diagnosis"""
    from datetime import datetime
    
    # Simple medication mapping (can be expanded)
    meds_map = {
        "Dandruff": "Ketoconazole shampoo 2%\nApply twice weekly",
        "रूसी": "कीटोकोनाज़ोल शैम्पू 2%\nसप्ताह में दो बार लगाएं",
        "Acne": "Benzoyl peroxide 5% cream\nApply daily at bedtime",
        "मुंहासे": "बेंज़ोयल पेरोक्साइड 5% क्रीम\nरोजाना सोने से पहले लगाएं"
    }
    
    # Get current date
    date = datetime.now().strftime("%d/%m/%Y")
    
    # Get appropriate medication based on language
    medication = meds_map.get(diagnosis, "Consult doctor for proper medication")
    
    return PRESCRIPTION_TEMPLATE.format(
        date=date,
        patient_name="[Patient Name]",
        diagnosis=diagnosis,
        medications=medication,
        instructions="Follow up in 2 weeks if condition persists"
    )

@lru_cache(maxsize=100)
def analyze_image_with_query(query, encoded_image, model="llama-3.2-11b-vision-preview"):
    """Analyze image with text query using GROQ's vision model with caching"""
    if not query or not encoded_image:
        raise ValueError("Missing required parameters")
        
    client = Groq(api_key=GROQ_API_KEY)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Vision analysis failed: {str(e)}")

# Validate GROQ API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("""
    ERROR: GROQ_API_KEY environment variable not set.
    Please either:
    1. Create a .env file in the project root with GROQ_API_KEY=your_key_here
    2. Or set it in your system environment variables
    
    You can get an API key from: https://console.groq.com/
    """)
    sys.exit(1)

def analyze_image(image_path):
    """Analyze image using computer vision"""
    try:
        from image_analysis import analyze_image_colors
        analysis = analyze_image_colors(image_path)
        return f"Image analysis results: Dominant colors are {', '.join(analysis['dominant_colors'])}"
    except Exception as e:
        raise ValueError(f"Image analysis failed: {str(e)}")

@lru_cache(maxsize=100)
def analyze_text_query(query, model="llama3-8b-8192", max_retries=3):
    """Process text queries with GROQ API with caching"""
    if not query or not isinstance(query, str):
        raise ValueError("Invalid query parameter")
        
    client = Groq(api_key=GROQ_API_KEY)
    
    messages = [
        {"role": "user", "content": query}
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192"  # Using available model
            )
            
            if not response.choices:
                raise ValueError("Empty response from API")
                
            return response.choices[0].message.content
            
        except GroqError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            raise ValueError(f"API request failed after {max_retries} attempts: {str(e)}")
            
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

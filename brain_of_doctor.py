import os
import base64
from dotenv import load_dotenv
from groq import Groq # type: ignore

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing. Please set it in the .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    """Convert an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_query(query, model, encoded_image):
    """
    Sends the image and query to Groq's vision model and returns the response.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ],
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return response.choices[0].message.content

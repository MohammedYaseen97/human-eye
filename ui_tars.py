import requests
import base64
from typing import List, Dict
from pathlib import Path
import time
import json
def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_ui_tars(prompt: str, image_path: str, base_url: str = "http://localhost:1234") -> Dict:
    """
    Query the UI-TARS model with a text prompt and image
    
    Args:
        prompt: Text prompt to send
        image_path: Path to the image file
        base_url: LM Studio server URL (default: http://172.17.0.1:1234 for WSL)
    
    Returns:
        JSON response from the API
    """
    # API endpoint
    url = f"{base_url}/v1/chat/completions"
    
    # Convert image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Construct the messages payload
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Prepare the full request payload
    payload = {
        "model": "lmstudio-community/UI-TARS-2B-SFT-GGUF",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    # Set headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Print request details for Postman testing
    print("\nPostman test details:")
    print(f"URL: {url}")
    print("\nHeaders:")
    for key, value in headers.items():
        print(f"{key}: {value}")
    print("\nRequest Body:")
    with open('request_body.txt', 'w') as f:
        f.write(json.dumps(payload, indent=2))
    print("Request body written to request_body.txt")
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    prompt = "What is this image?"
    image_path = "ui_cropped.jpg"
    result = query_ui_tars(prompt, image_path)
    print("Response:", result)


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import os
import io

load_dotenv()

app = FastAPI()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is not set in the .env file")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

class TextToImageRequest(BaseModel):
    prompt: str

def query(payload):
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.content

@app.post("/generate-image/")
async def generate_image(request: TextToImageRequest):
    try:
        payload = {
            "inputs": request.prompt,
            "options": {"use_gpu": True}
        }
        image_data = query(payload)
        
        # Create a BytesIO stream from the image data
        image_stream = io.BytesIO(image_data)
        
        # Return the image as a StreamingResponse
        return StreamingResponse(image_stream, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

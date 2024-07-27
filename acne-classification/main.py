from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os
from tenacity import retry, wait_fixed, stop_after_attempt

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Hugging Face API settings
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/imfarzanansari/skintelligent-acne"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def query(image_str):
    payload = {
        "inputs": image_str
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

@app.post("/classify-acne")
async def classify_acne(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        result = query(img_str)
        return JSONResponse(content={"classification": result})
    except requests.exceptions.HTTPError as http_err:
        raise HTTPException(status_code=503, detail=f"HTTP error occurred: {http_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

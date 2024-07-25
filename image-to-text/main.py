from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def query(payload):
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    return response.json()


@app.post("/upload-image")
async def unconditional_caption(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        payload = {
            "inputs": {
                "image": img_str
            }
        }
        result = query(payload)
        return JSONResponse(content={"caption": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

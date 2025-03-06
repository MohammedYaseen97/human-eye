import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import base64
import json
from models.predictor import UIPredictor
from models.ui_attention_predictor import Platform
from pydantic import BaseModel
from typing import Optional
import traceback

# Load environment variables
load_dotenv()

# Add models directory to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

app = FastAPI(title="UI Attention Predictor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local frontend
        "http://localhost:8000",  # Local backend
        "https://your-vercel-app.vercel.app",  # Production frontend (when you deploy)
        "https://your-custom-domain.com"        # If you have a custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
predictor = UIPredictor()

class PredictionRequest(BaseModel):
    age: int
    platform: str
    task: str
    tech_saviness: int
    debug: Optional[bool] = False

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/predict")
async def predict_attention(
    file: UploadFile = File(...),
    age: int = 25,
    task: str = "find settings",
    tech_saviness: int = 3,
    debug: bool = False
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction without platform parameter
        async def generate():
            try:
                for result in predictor.predict(
                    image=image,
                    age=age,
                    task=task,
                    tech_saviness=tech_saviness,
                    debug=debug
                ):
                    if isinstance(result, dict):
                        yield f"data: {json.dumps(result)}\n\n"
            except Exception as e:
                error_data = {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "UI Attention Predictor API is running"} 
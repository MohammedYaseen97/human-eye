import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from models.predictor import UIPredictor
from models.ui_attention_predictor import Platform
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add models directory to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

app = FastAPI(title="UI Attention Predictor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
predictor = UIPredictor()

@app.post("/predict")
async def predict_attention(
    file: UploadFile = File(...),
    age: int = 25,
    task: str = "find settings",
    tech_saviness: int = 3,
    platform: str = "desktop",
    debug: bool = False
):
    logger.info(f"Received request - Platform: {platform}, Age: {age}, Task: {task}")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.debug("Image loaded successfully")
        
        final_result = None
        # Just iterate through and keep the last one
        for result in predictor.predict(
            image=image,
            age=age,
            platform=Platform(platform),
            task=task,
            tech_saviness=tech_saviness,
            debug=debug
        ):
            final_result = result
        
        return JSONResponse(content=final_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "UI Attention Predictor API is running"} 
import base64
import mimetypes
import json
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from anthropic import Anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Image Detector")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)
client = Anthropic(api_key=ANTHROPIC_API_KEY)

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif"
}

def get_mime_type(filename: str) -> str:
    """Get MIME type from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}")
    return mime_type

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    # Fixed: Added encoding='utf-8' to handle Unicode characters
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Analyze uploaded image to determine if it's AI-generated or real"""
    try:
        # Validate file type
        mime_type = get_mime_type(file.filename)
        
        # Read and encode image
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        
        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            temperature=0,
            system="You are an AI and Real image identifier. Respond ONLY with valid JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """
                            Your task is to determine whether the image is AI-generated or captured by a real camera.
                            If the skin tone appears overly uniform, smooth, and consistent across different areas, classify the image as AI-generated.
                            If the skin shows natural imperfections such as dark spots, blemishes, texture variations, or uneven tone, classify it as a real image.
                            Respond ONLY with a JSON object in this exact format:
                            {
                                "classification": "AI-Generated" or "Real Image",
                                "confidence": 85,
                                "reasoning": "Brief explanation of your analysis",
                                "details": ["Key observation 1", "Key observation 2", "Key observation 3"]
                            }
                            """
                        }
                    ]
                }
            ]
        )
        
        # Parse response
        response_text = response.content[0].text
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback parsing
            result = {
                "classification": "Analysis Complete",
                "confidence": 75,
                "reasoning": response_text,
                "details": ["Analysis provided above"]
            }
        
        return JSONResponse(content={
            "success": True,
            "result": result
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
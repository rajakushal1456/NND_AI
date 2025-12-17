import base64
import mimetypes
import json
from typing import Dict, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import io

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

SUPPORTED_TEXT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.ms-powerpoint",  # .ppt
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "text/plain"  # .txt
}

class TextAnalysisRequest(BaseModel):
    text: str

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap to maintain context.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 chars
            sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if any(text[i:i+2] == ending for ending in sentence_endings):
                    end = i + 2
                    break
        
        chunk = text[start:end]
        chunks.append(chunk.strip())
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else end
    
    return chunks

def split_into_segments(text: str) -> List[Dict[str, any]]:
    """
    Split text into segments (sentences/paragraphs) for detailed analysis.
    
    Args:
        text: The text to segment
    
    Returns:
        List of segments with their text and position
    """
    import re
    
    # For very short text, return as single segment
    if len(text) < 100:
        return [{
            "text": text,
            "start": 0,
            "end": len(text)
        }]
    
    # Split by paragraphs first (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    segments = []
    current_pos = 0
    
    for para in paragraphs:
        if not para.strip():
            current_pos += len(para) + 2  # +2 for the double newline
            continue
        
        # Split paragraph into sentences
        sentences = re.split(r'([.!?]\s+)', para)
        
        current_segment = ""
        segment_start = current_pos
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                
                current_segment += sentence
                
                # Create segment every 2-3 sentences or if segment is long enough (200-400 chars)
                if len(current_segment) > 250 or (i + 2 >= len(sentences)):
                    if current_segment.strip():
                        segment_text = current_segment.strip()
                        segments.append({
                            "text": segment_text,
                            "start": segment_start,
                            "end": segment_start + len(segment_text)
                        })
                        segment_start += len(current_segment)
                        current_segment = ""
        
        # Handle remaining text in paragraph
        if current_segment.strip():
            segment_text = current_segment.strip()
            segments.append({
                "text": segment_text,
                "start": segment_start,
                "end": segment_start + len(segment_text)
            })
        
        current_pos += len(para) + 2
    
    # If no segments created, split by sentences only
    if not segments:
        sentences = re.findall(r'[^.!?]+[.!?]\s*', text)
        current_pos = 0
        current_segment = ""
        segment_start = 0
        
        for sentence in sentences:
            current_segment += sentence
            if len(current_segment) > 200:
                if current_segment.strip():
                    segment_text = current_segment.strip()
                    segments.append({
                        "text": segment_text,
                        "start": segment_start,
                        "end": segment_start + len(segment_text)
                    })
                    segment_start += len(current_segment)
                    current_segment = ""
        
        if current_segment.strip():
            segment_text = current_segment.strip()
            segments.append({
                "text": segment_text,
                "start": segment_start,
                "end": segment_start + len(segment_text)
            })
    
    # Final fallback: return whole text as single segment
    if not segments:
        segments.append({
            "text": text,
            "start": 0,
            "end": len(text)
        })
    
    return segments

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
                            if(person in image){
                            If the skin tone appears overly uniform, smooth, and consistent across different areas, classify the image as AI-generated.
                            If the skin shows natural imperfections such as dark spots, blemishes, texture variations, or uneven tone, classify it as a real image.
                            }
                            else{
                               If the composition is unrealistically perfect with flawless symmetry, overly uniform lighting throughout all interior spaces, and an idealized presentation that lacks the natural imperfections of real photography. The textures, reflections, and material rendering show the characteristic smoothness and consistency typical of AI-generated imagery.  
                            }
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

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> JSONResponse:
    """Extract text from uploaded document (PDF, DOCX, PPT, TXT)"""
    try:
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        contents = await file.read()
        
        text = ""
        
        if file_ext == 'txt':
            text = contents.decode('utf-8', errors='ignore')
        elif file_ext == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                # Fallback: try pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(contents)) as pdf:
                        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                except ImportError:
                    raise HTTPException(status_code=500, detail="PDF extraction library not installed. Please install PyPDF2 or pdfplumber.")
        elif file_ext == 'docx':
            try:
                from docx import Document
                doc = Document(io.BytesIO(contents))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                raise HTTPException(status_code=500, detail="DOCX extraction library not installed. Please install python-docx.")
        elif file_ext in ['ppt', 'pptx']:
            try:
                from pptx import Presentation
                prs = Presentation(io.BytesIO(contents))
                text_parts = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text_parts.append(shape.text)
                text = "\n".join(text_parts)
            except ImportError:
                raise HTTPException(status_code=500, detail="PPT extraction library not installed. Please install python-pptx.")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        return JSONResponse(content={
            "success": True,
            "text": text
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

async def analyze_text_segment(segment: str) -> Dict:
    """Analyze a single text segment"""
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=800,
        temperature=0,
        system="You are an AI text detector. Analyze text segments to determine if they're AI-generated or human-written. Respond ONLY with valid JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Analyze this text segment and determine if it's AI-generated or human-written.
                
                Consider:
                - AI patterns: overly formal, repetitive structures, lack of personal voice, generic phrasing
                - Human patterns: varied sentence structure, personal touches, natural flow, specific details
                
                Text segment:
                {segment}
                
                Respond ONLY with a JSON object:
                {{
                    "classification": "AI-Generated", "Slightly-AI", or "Human-Written",
                    "confidence": 85
                }}
                
                Use "Slightly-AI" if the text shows some AI characteristics but also has human elements.
                """
            }
        ]
    )
    
    response_text = response.content[0].text
    
    # Try to extract JSON if it's wrapped in markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "classification": "Human-Written",
            "confidence": 50
        }

async def analyze_text_chunk(chunk: str, chunk_num: int, total_chunks: int) -> Dict:
    """Analyze a single chunk of text"""
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        temperature=0,
        system="You are an AI text detector. Analyze text to determine if it's AI-generated or human-written. Respond ONLY with valid JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Your task is to determine whether the following text chunk (chunk {chunk_num} of {total_chunks}) is AI-generated or human-written.
                
                Analyze the text for:
                - Writing patterns typical of AI (overly formal, repetitive structures, lack of personal voice)
                - Natural human writing characteristics (varied sentence structure, personal touches, natural flow)
                - Consistency and coherence
                - Use of specific details and authentic experiences
                
                Text chunk to analyze:
                {chunk}
                
                Respond ONLY with a JSON object in this exact format:
                {{
                    "classification": "AI-Generated" or "Human-Written",
                    "confidence": 85,
                    "reasoning": "Brief explanation of your analysis",
                    "details": ["Key observation 1", "Key observation 2", "Key observation 3"]
                }}
                """
            }
        ]
    )
    
    response_text = response.content[0].text
    
    # Try to extract JSON if it's wrapped in markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "classification": "Analysis Complete",
            "confidence": 75,
            "reasoning": response_text,
            "details": ["Analysis provided above"]
        }

def aggregate_chunk_results(chunk_results: List[Dict]) -> Dict:
    """Aggregate results from multiple chunks into a single result"""
    if not chunk_results:
        return {
            "classification": "Analysis Complete",
            "confidence": 50,
            "reasoning": "No chunks analyzed",
            "details": []
        }
    
    # Count classifications
    ai_count = sum(1 for r in chunk_results if "ai" in r.get("classification", "").lower())
    human_count = len(chunk_results) - ai_count
    
    # Calculate average confidence
    avg_confidence = sum(r.get("confidence", 50) for r in chunk_results) / len(chunk_results)
    
    # Determine overall classification
    if ai_count > human_count:
        classification = "AI-Generated"
        confidence = int((ai_count / len(chunk_results)) * avg_confidence)
    elif human_count > ai_count:
        classification = "Human-Written"
        confidence = int((human_count / len(chunk_results)) * avg_confidence)
    else:
        classification = "Mixed/Uncertain"
        confidence = int(avg_confidence)
    
    # Aggregate reasoning and details
    all_reasonings = [r.get("reasoning", "") for r in chunk_results if r.get("reasoning")]
    all_details = []
    for r in chunk_results:
        all_details.extend(r.get("details", []))
    
    # Remove duplicates and limit details
    unique_details = list(dict.fromkeys(all_details))[:5]  # Keep first 5 unique details
    
    reasoning = f"Analyzed {len(chunk_results)} chunk(s). " + " ".join(all_reasonings[:2])  # Combine first 2 reasonings
    
    return {
        "classification": classification,
        "confidence": min(100, max(0, confidence)),  # Ensure 0-100 range
        "reasoning": reasoning,
        "details": unique_details
    }

@app.post("/analyze-text")
async def analyze_text(request: TextAnalysisRequest) -> JSONResponse:
    """Analyze text to determine if it's AI-generated or human-written with segment-level highlighting"""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        text = request.text.strip()
        
        # Split text into segments for detailed analysis
        segments = split_into_segments(text)
        
        # Analyze each segment
        segment_results = []
        for segment_data in segments:
            segment_text = segment_data["text"]
            segment_analysis = await analyze_text_segment(segment_text)
            segment_results.append({
                "text": segment_text,
                "start": segment_data["start"],
                "end": segment_data["end"],
                "classification": segment_analysis.get("classification", "Human-Written"),
                "confidence": segment_analysis.get("confidence", 50)
            })
        
        # Calculate percentages based on text length
        total_length = len(text)
        ai_generated_length = 0
        slightly_ai_length = 0
        human_written_length = 0
        
        for segment in segment_results:
            segment_length = segment["end"] - segment["start"]
            classification = segment["classification"].lower()
            
            if "ai-generated" in classification or "ai generated" in classification:
                ai_generated_length += segment_length
            elif "slightly" in classification or "slightly-ai" in classification:
                slightly_ai_length += segment_length
            else:
                human_written_length += segment_length
        
        # Calculate percentages
        ai_percentage = round((ai_generated_length / total_length * 100), 2) if total_length > 0 else 0
        slightly_ai_percentage = round((slightly_ai_length / total_length * 100), 2) if total_length > 0 else 0
        human_percentage = round((human_written_length / total_length * 100), 2) if total_length > 0 else 0
        
        # Total AI percentage (AI + Slightly AI)
        total_ai_percentage = round(ai_percentage + slightly_ai_percentage, 2)
        
        # Also get overall analysis for summary
        chunks = chunk_text(text, chunk_size=8000, overlap=200)
        
        if len(chunks) == 1:
            overall_result = await analyze_text_chunk(chunks[0], 1, 1)
        else:
            chunk_results = []
            for i, chunk in enumerate(chunks, 1):
                chunk_result = await analyze_text_chunk(chunk, i, len(chunks))
                chunk_results.append(chunk_result)
            overall_result = aggregate_chunk_results(chunk_results)
        
        return JSONResponse(content={
            "success": True,
            "result": overall_result,
            "original_text": text,
            "segments": segment_results,
            "statistics": {
                "total_length": total_length,
                "ai_generated_length": ai_generated_length,
                "slightly_ai_length": slightly_ai_length,
                "human_written_length": human_written_length,
                "ai_percentage": ai_percentage,
                "slightly_ai_percentage": slightly_ai_percentage,
                "human_percentage": human_percentage,
                "total_ai_percentage": total_ai_percentage
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
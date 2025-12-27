from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer
import time
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import SarcasmVibeModel

app = FastAPI(title="Sarcasm Vibe Detector API")

# Mock model for foundation code (in real scenario, load weights)
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SarcasmVibeModel(MODEL_NAME)
model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    vibe_score: float
    latency_ms: float

LABELS = ["Neutral", "Sincere", "Sarcastic"]

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()
    
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)
        confidence, index = torch.max(probs, dim=-1)
        
    latency = (time.time() - start_time) * 1000
    
    # Vibe score calculation (simplified)
    vibe_score = probs[0][2].item() if index == 2 else (1 - probs[0][0].item())

    return PredictResponse(
        label=LABELS[index.item()],
        confidence=confidence.item(),
        vibe_score=vibe_score,
        latency_ms=latency
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

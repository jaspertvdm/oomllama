"""
OomLlama API Server
====================

REST API for .oom format LLM inference.

Endpoints:
- POST /generate - Generate text from prompt
- POST /chat - Chat completion
- GET /models - List available models
- GET /info - Model info
- GET /health - Health check
"""

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oomllama import OomLlama, list_models, version

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "humotica-32b")
MODEL_PATH = os.environ.get("MODEL_PATH")
GPU_ID = os.environ.get("GPU_ID")

# Initialize
app = FastAPI(
    title="OomLlama API",
    description="Efficient LLM inference with .oom format - 2x smaller than GGUF",
    version="0.8.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instance (lazy loaded)
_model: Optional[OomLlama] = None

def get_model() -> OomLlama:
    global _model
    if _model is None:
        gpu = int(GPU_ID) if GPU_ID else None
        _model = OomLlama(MODEL_NAME, model_path=MODEL_PATH, gpu=gpu)
    return _model


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class GenerateResponse(BaseModel):
    response: str
    model: str
    format: str = "oom"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    response: str
    model: str


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": version() if callable(version) else "0.8.0",
        "format": "oom",
        "model": MODEL_NAME
    }


@app.get("/models")
async def models():
    """List available models."""
    return {
        "models": list_models(),
        "current": MODEL_NAME,
        "format": "oom"
    }


@app.get("/info")
async def info():
    """Model information."""
    return {
        "name": MODEL_NAME,
        "format": "oom",
        "quantization": "Q2/Q4/Q8",
        "description": "Quantized weights with per-block scale/min (256 block size)",
        "features": [
            "Q2/Q4/Q8 Quantization with F32 norms",
            "SafeTensors + GGUF converters",
            "Lazy Layer Loading",
            "Interleaved RoPE (Qwen support)",
            "GPU acceleration via CUDA/Candle"
        ]
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    try:
        model = get_model()
        model.set_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        response = model.generate(request.prompt)
        return GenerateResponse(
            response=response,
            model=MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat completion."""
    try:
        model = get_model()
        messages = [(m.role, m.content) for m in request.messages]
        response = model.chat(messages, max_tokens=request.max_tokens)
        return ChatResponse(
            response=response,
            model=MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "service": "OomLlama API",
        "version": "0.8.0",
        "description": "Efficient LLM inference with .oom format",
        "format": {
            "name": "OOM (OomLlama Model)",
            "quantization": "Q2 (2-bit)",
            "compression": "~2x smaller than GGUF Q4"
        },
        "endpoints": {
            "generate": "POST /generate - Generate text",
            "chat": "POST /chat - Chat completion",
            "models": "GET /models - List models",
            "info": "GET /info - Model info",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

#!/usr/bin/env python3
import os, time, uuid, logging
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_CTX = int(os.getenv("N_CTX", "2048"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
llm: Optional[Llama] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    logger.info(f"🔄 Carregando modelo: {os.path.basename(MODEL_PATH)}")
    start = time.time()
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
    logger.info(f"✅ Modelo carregado em {time.time()-start:.1f}s")
    yield
    logger.info("🔄 Desligando...")

app = FastAPI(title="TinyLlama Chat API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "tinyllama"
    messages: List[Message]
    max_tokens: int = Field(default=256, le=MAX_TOKENS)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

@app.get("/")
def root(): return {"name": "TinyLlama Chat API", "status": "running", "docs": "/docs"}

@app.get("/health")
def health(): return {"status": "healthy", "model": os.path.basename(MODEL_PATH), "device": "CPU"}

@app.get("/v1/models")
def list_models(): return {"object": "list", "data": [{"id": "tinyllama-1.1b-chat-v1.0", "object": "model", "created": int(time.time()), "owned_by": "llama.cpp"}]}

def format_prompt(messages: List[Message]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system": prompt += f"<|system|>\n{msg.content}</s>\n"
        elif msg.role == "user": prompt += f"<|user|>\n{msg.content}</s>\n"
        elif msg.role == "assistant": prompt += f"<|assistant|>\n{msg.content}</s>\n"
    return prompt + "<|assistant|>\n"

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    if llm is None: raise HTTPException(503, "Model not loaded")
    prompt = format_prompt(req.messages)
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    if req.stream:
        async def stream_gen():
            for token in llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature, stream=True):
                chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": req.model, "choices": [{"index": 0, "delta": {"content": token["choices"][0]["text"]}, "finish_reason": None}]}
                yield f" {chunk}\n\n"
            yield " [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        output = llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        content = output["choices"][0]["text"]
        return ChatResponse(id=response_id, created=created, model=req.model, choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], usage={"prompt_tokens": output["usage"]["prompt_tokens"], "completion_tokens": output["usage"]["completion_tokens"], "total_tokens": output["usage"]["total_tokens"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, log_level="info")

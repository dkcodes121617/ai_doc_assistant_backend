import os
import threading
import time
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.security import limiter
from app.core.config import settings
from app.routers import upload, chat, health

def ping_loop():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        print("[KeepAlive] RENDER_EXTERNAL_URL not set. Skipping self-ping (only needed on Render).")
        return
    
    url = url.rstrip("/") + "/health"
    print(f"[KeepAlive] Starting self-ping to {url} every 10 minutes.")
    
    while True:
        time.sleep(600)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Render-KeepAlive'})
            with urllib.request.urlopen(req, timeout=10) as response:
                print(f"[KeepAlive] Self-ping successful: {response.getcode()}")
        except Exception as e:
            print(f"[KeepAlive] Self-ping failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()
    yield

app = FastAPI(title="RAG AI Assistant API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(chat.router)

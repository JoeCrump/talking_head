"""
FastAPI application for Auto Video Summarizer.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.routers import videos

# Create output directory if it doesn't exist
os.makedirs("static/output", exist_ok=True)

app = FastAPI(
    title="Auto Video Summarizer API",
    description="API for creating short-form videos from longer content with captions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(videos.router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Auto Video Summarizer API",
        "docs_url": "/docs",
        "upload_video": "/videos/upload",
    }
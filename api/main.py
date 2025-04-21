"""
Main FastAPI application.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Video Summarizer API",
    description="API for creating short-form videos from longer content",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
os.makedirs("static/output", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import and include API routes
from api.routers.videos import router as videos_router
app.include_router(videos_router, prefix="/api/videos")

# Import and initialize storage
from api.storage import initialize_storage

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    try:
        initialize_storage()
        logger.info("Storage initialized")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Video Summarizer API"}
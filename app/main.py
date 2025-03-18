import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')

# Ensure the correct path is added
sys.path.append(PROJECT_ROOT)

# Local imports
from app.models import PredictionInput, CollegeDetailInput
from app.utils import (
    load_data, 
    get_unique_branches, 
    predict_preferences, 
    get_college_details
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(
    title="JOSAA College Predictor",
    description="Advanced College Prediction Platform",
    version="1.2.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static file serving
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Perform startup tasks"""
    try:
        load_data()
        logger.info("Data loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load data on startup: {e}")

# API Endpoints
@app.get("/api/branches")
def get_branches():
    """Retrieve unique academic branches"""
    try:
        branches = get_unique_branches()
        logger.info(f"Retrieved {len(branches)} branches")
        return branches
    except Exception as e:
        logger.error(f"Error retrieving branches: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve branches")

@app.post("/api/predict")
def predict(input: PredictionInput):
    """Predict college preferences based on input parameters"""
    try:
        result, _, plot = predict_preferences(
            input.jee_rank,
            input.category,
            input.college_type,
            input.preferred_branch,
            input.round_no,
            input.min_probability
        )
        
        if "Error" in result.columns:
            return JSONResponse(
                status_code=400, 
                content={"message": "No predictions available"}
            )
        
        preferences = result.to_dict(orient='records')
        plot_data = plot.to_dict() if plot else None

        return {
            "preferences": preferences, 
            "plot_data": plot_data
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/college-details")
def college_details(input: CollegeDetailInput):
    """Retrieve detailed information about a specific college"""
    try:
        details = get_college_details(input.institute, input.branch)
        if "error" in details:
            raise HTTPException(status_code=404, detail=details["error"])
        return details
    except Exception as e:
        logger.error(f"Error fetching college details: {e}")
        raise HTTPException(status_code=404, detail="College details not found")

@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serve the main index.html"""
    try:
        with open(os.path.join(TEMPLATES_DIR, "index.html"), "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Application Frontend Not Found</h1>", status_code=404)

# Uvicorn Server Runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

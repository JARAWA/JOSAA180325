from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import logging
from .models import PredictionInput, CollegeDetailInput
from .utils import (
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files and templates
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    try:
        load_data()
        logger.info("Data loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load data on startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main index page"""
    try:
        with open(os.path.join(current_dir, "templates", "index.html")) as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return HTMLResponse(content="<h1>Error loading page</h1>")

def get_unique_branches() -> list:
    """
    Retrieve unique academic branches with enhanced error checking
    
    Returns:
        list: Sorted list of unique branches
    """
    global JOSAA_DATA
    try:
        if JOSAA_DATA is None:
            logger.warning("JOSAA_DATA is None, attempting to load data...")
            JOSAA_DATA = load_data()
            
        # Debug information about the DataFrame
        logger.info(f"DataFrame shape: {JOSAA_DATA.shape}")
        logger.info(f"DataFrame columns: {JOSAA_DATA.columns.tolist()}")
        
        if 'Academic Program Name' not in JOSAA_DATA.columns:
            logger.error("'Academic Program Name' column not found in DataFrame")
            return {
                "status": "error",
                "message": "Academic Program Name column missing",
                "branches": ["All"]
            }
        
        # Get unique branches and clean them
        branches = JOSAA_DATA['Academic Program Name'].dropna()
        logger.info(f"Number of non-null branches before cleaning: {len(branches)}")
        
        branches = branches.apply(lambda x: str(x).strip().lower())
        unique_branches = sorted(list(set(branches)))
        
        logger.info(f"Number of unique branches found: {len(unique_branches)}")
        logger.debug(f"Unique branches: {unique_branches[:5]}...")  # Show first 5 branches
        
        # Add "All" option at the beginning
        unique_branches = ["All"] + unique_branches
        
        return {
            "status": "success",
            "message": f"Successfully retrieved {len(unique_branches)} branches",
            "branches": unique_branches
        }
        
    except Exception as e:
        logger.error(f"Error in get_unique_branches: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving branches: {str(e)}",
            "branches": ["All"]
        }

@app.post("/api/predict")
async def predict(input: PredictionInput):
    """
    Predict college preferences based on input parameters
    
    Args:
        input (PredictionInput): Input parameters for prediction
    
    Returns:
        Dict containing predictions and visualization data
    """
    try:
        result, plot = predict_preferences(
            input.jee_rank,
            input.category,
            input.college_type,
            input.preferred_branch,
            input.round_no,
            input.min_probability
        )
        
        if result.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No predictions available for given criteria"}
            )
        
        return {
            "predictions": result.to_dict(orient='records'),
            "plot_data": plot.to_dict() if plot else None
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/college-details")
async def college_details(input: CollegeDetailInput):
    """
    Retrieve detailed information about a specific college
    
    Args:
        input (CollegeDetailInput): College and branch information
    
    Returns:
        Dict containing college details
    """
    try:
        details = get_college_details(input.institute, input.branch)
        if "error" in details:
            raise HTTPException(status_code=404, detail=details["error"])
        return details
    except Exception as e:
        logger.error(f"Error in college_details endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )

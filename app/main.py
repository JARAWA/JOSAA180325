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
    get_college_details,
    JOSAA_DATA
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

@app.get("/api/branches")
async def get_branches():
    """
    Retrieve unique academic branches
    """
    try:
        branches = get_unique_branches()
        logger.info(f"API request for branches. Found {len(branches)} branches")
        
        if not branches or len(branches) <= 1:
            logger.warning("No branches found or only 'All' was returned")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "warning",
                    "message": "Using fallback branch list",
                    "branches": ["All", "computer science and engineering", 
                               "electrical engineering", "mechanical engineering", 
                               "civil engineering"]
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Successfully retrieved {len(branches)} branches",
                "branches": branches
            }
        )
    except Exception as e:
        logger.error(f"Error in get_branches endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "branches": []
            }
        )

@app.post("/api/predict")
async def predict(input: PredictionInput):
    """
    Predict college preferences based on input parameters
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
        
        return JSONResponse(
            status_code=200,
            content={
                "predictions": result.to_dict(orient='records'),
                "plot_data": plot.to_dict() if plot else None
            }
        )
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/college-details")
async def college_details(input: CollegeDetailInput):
    """
    Retrieve detailed information about a specific college
    """
    try:
        details = get_college_details(input.institute, input.branch)
        if "error" in details:
            return JSONResponse(
                status_code=404,
                content={"error": details["error"]}
            )
        return JSONResponse(status_code=200, content=details)
    except Exception as e:
        logger.error(f"Error in college_details endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "data_loaded": JOSAA_DATA is not None,
            "rows_count": len(JOSAA_DATA) if JOSAA_DATA is not None else 0
        }
    )

@app.get("/api/test-branches")
async def test_branches():
    """Test endpoint for branch data"""
    try:
        branches = get_unique_branches()
        return JSONResponse(
            status_code=200,
            content={
                "total_branches": len(branches),
                "sample_branches": branches[:5],
                "csv_loaded": JOSAA_DATA is not None,
                "csv_rows": len(JOSAA_DATA) if JOSAA_DATA is not None else 0
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )

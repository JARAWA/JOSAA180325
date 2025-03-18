from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .models import PredictionInput, PredictionOutput
from .utils import load_data, get_unique_branches, predict_preferences

app = FastAPI(title="JOSAA Predictor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/api/branches", response_model=list)
def get_branches():
    return get_unique_branches()

@app.post("/api/predict", response_model=PredictionOutput)
def predict(input: PredictionInput):
    result, _, plot = predict_preferences(
        input.jee_rank,
        input.category,
        input.college_type,
        input.preferred_branch,
        input.round_no,
        input.min_probability
    )
    if "Error" in result.columns:
        return {"preferences": [], "plot_data": None}
    
    preferences = result.to_dict(orient='records')
    plot_data = plot.to_dict() if plot else None

    return {"preferences": preferences, "plot_data": plot_data}

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("templates/index.html") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

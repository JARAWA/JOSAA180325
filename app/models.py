from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    jee_rank: int = Field(..., gt=0, description="JEE Main Rank")
    category: str = Field(..., description="Category (e.g., OPEN, SC, ST)")
    college_type: str = Field(..., description="College Type (e.g., IIT, NIT)")
    preferred_branch: str = Field(..., description="Preferred Branch")
    round_no: int = Field(..., ge=1, le=7, description="Counseling Round Number")
    min_probability: float = Field(..., ge=0, le=100, description="Minimum Probability Threshold")

class CollegeDetailInput(BaseModel):
    institute: str = Field(..., description="Institute Name")
    branch: str = Field(..., description="Branch Name")

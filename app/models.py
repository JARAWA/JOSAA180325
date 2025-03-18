from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    jee_rank: int = Field(..., gt=0, le=1000000, description="JEE Rank")
    category: str = Field(default="GENERAL", description="Reservation Category")
    college_type: str = Field(default="ALL", description="Type of College")
    preferred_branch: str = Field(default="ALL", description="Preferred Branch")
    round_no: int = Field(default=1, ge=1, le=6, description="Counseling Round")
    min_probability: float = Field(default=40.0, ge=0, le=100, description="Minimum Probability Threshold")

class CollegeDetailInput(BaseModel):
    institute: str
    branch: str

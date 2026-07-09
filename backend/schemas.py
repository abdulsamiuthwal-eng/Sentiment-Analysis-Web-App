from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., description="The raw input text to analyze")

class PredictionResponse(BaseModel):
    text: str = Field(..., description="The original input text")
    prediction: str = Field(..., description="The predicted sentiment (Positive, Neutral, or Negative)")
    confidence: Optional[str] = Field(None, description="The confidence score of the prediction, if available (e.g., '94.82%')")

from pydantic import BaseModel
from typing import List, Optional

class DatasetInfo(BaseModel):
    filename: str
    rows: int
    columns: int

class AlgorithmRecommendation(BaseModel):
    algorithm: str
    confidence: float
    rank: int
    explanation: str
    characteristics: dict

class PredictionResponse(BaseModel):
    success: bool
    recommendations: List[dict]
    visualization_url: Optional[str] = None
    dataset_info: Optional[DatasetInfo] = None

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.predictor import MetaLearningPredictor
from core.visualizer import generate_visualization
from api.models import PredictionResponse, DatasetInfo

app = FastAPI(
    title="Meta-Learning Algorithm Recommender API",
    description="API for ML algorithm recommendations using meta-learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = MetaLearningPredictor()

@app.get("/")
async def root():
    return {
        "message": "Meta-Learning Algorithm Recommender API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor.is_loaded()}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_algorithms(file: UploadFile = File(...)):
    """
    Upload a CSV dataset and get algorithm recommendations
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    # Save uploaded file temporarily
    temp_path = Path(settings.TEMP_DIR) / file.filename
    temp_path.parent.mkdir(exist_ok=True)
    
    try:
        # Save file
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Get predictions
        recommendations = predictor.predict(str(temp_path))
        
        # Generate visualization
        viz_path = generate_visualization(recommendations)
        
        return PredictionResponse(
            success=True,
            recommendations=recommendations,
            visualization_url=f"/api/visualization/{viz_path.name}",
            dataset_info=DatasetInfo(
                filename=file.filename,
                rows=recommendations[0].get('dataset_rows', 0),
                columns=recommendations[0].get('dataset_cols', 0)
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve generated visualization"""
    viz_path = Path("temp") / filename
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(viz_path)

@app.get("/api/algorithms")
async def list_algorithms():
    """Get list of supported algorithms"""
    return {
        "algorithms": predictor.get_supported_algorithms()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

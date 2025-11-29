from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 10000
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:3000"]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    TEMP_DIR: str = "./temp"
    
    # Model Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR.parent / "data"
    MODEL_PATH: Path = DATA_DIR / "mlknn_model.pkl"
    FEATURES_PATH: Path = DATA_DIR / "selected_features.pkl"
    IMPUTER_PATH: Path = DATA_DIR / "imputer.pkl"
    
    # OpenAI (optional)
    OPENAI_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()

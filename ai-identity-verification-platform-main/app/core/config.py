"""Centralized application settings, loaded from environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Identity Verification Platform"
    api_v1_prefix: str = "/api/v1"
    face_match_threshold: float = 0.45  # cosine similarity threshold for ArcFace
    log_level: str = "INFO"
    gemini_api_key: str = ""  # Loaded safely from .env or environment variables
    
    class Config:
        env_file = ".env"


settings = Settings()
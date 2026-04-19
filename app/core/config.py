from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Driven DNA Mutation Impact Platform"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # ESM Model
    ESM_MODEL_NAME: str = "esm2_t6_8M_UR50D"
    
    # ESMFold API
    ESMFOLD_API_URL: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    # Supabase (optional — set via env vars)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None
    
    # Database mode
    USE_SUPABASE: bool = False

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as biology_router
from app.api.structure_routes import router as structure_router
from app.api.whatif_routes import router as whatif_router
from app.api.auth_routes import router as auth_router
from app.api.experiment_routes import router as experiment_router
from app.core.config import settings

from contextlib import asynccontextmanager
from app.db.session import engine, Base
import app.db.models  # to ensure Base catches the tables
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Pre-load AI models
    logger.info("Pre-loading AI models...")
    
    try:
        from app.services.esm_service import ensure_model_loaded
        ensure_model_loaded()
        logger.info("✓ ESM-2 model loaded")
    except Exception as e:
        logger.warning(f"✗ ESM-2 model failed to load (will use fallback): {e}")
    
    try:
        from app.services.classifier_service import ensure_classifier_loaded
        ensure_classifier_loaded()
        logger.info("✓ Pathogenicity classifier loaded")
    except Exception as e:
        logger.warning(f"✗ Classifier failed to load (will use fallback): {e}")
    
    logger.info("AI pipeline ready.")
    
    yield
    # Cleanup here if needed
    
def start_application():
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"/openapi.json",
        lifespan=lifespan
    )

    # Parse allowed origins from env, fallback to wildcard for local dev
    raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
    if raw_origins == "*":
        allow_origins = ["*"]
        allow_credentials = False
    else:
        allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        allow_credentials = True

    # Set up CORS for the frontend React application
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Mount routers
    app.include_router(biology_router, prefix="/api/v1/biology", tags=["Biology Execution Engine"])
    app.include_router(structure_router, prefix="/api/v1/structure", tags=["Protein Structure"])
    app.include_router(whatif_router, prefix="/api/v1/whatif", tags=["What-If Simulation"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(experiment_router, prefix="/api/v1/experiments", tags=["Experiments"])

    return app

app = start_application()

@app.get("/")
def read_root():
    return {"status": "AI-Driven DNA Simulation Engine Active", "version": settings.VERSION}

from fastapi import FastAPI
from src.api.v1.router import router as api_router
from src.core.config import settings

app = FastAPI(title="Datathon AI Recruiter")

app.include_router(api_router)

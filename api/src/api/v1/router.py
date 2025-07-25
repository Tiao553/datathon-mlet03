from fastapi import APIRouter
from src.api.v1.endpoints import predict

router = APIRouter()


router.include_router(predict.router, prefix="/predict", tags=["predict"])

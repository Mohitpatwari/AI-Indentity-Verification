from fastapi import APIRouter
from app.api.v1.endpoints import verify
from app.api.v1.endpoints import verify, history

api_router = APIRouter()
api_router.include_router(verify.router, tags=["verification"])
api_router.include_router(history.router, tags=["history"])
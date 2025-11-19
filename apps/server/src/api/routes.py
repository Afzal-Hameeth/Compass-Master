from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "ok"},
        status_code=200
    )
    
@router.get("/{full_path:path}")
async def catch_all(full_path: str):
    raise HTTPException(
        status_code=404,
        detail=f"Route '{full_path}' not found."
    )
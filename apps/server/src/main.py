import sys
import os
from pathlib import Path


src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from env import env
from tortoise.contrib.fastapi import register_tortoise

app = FastAPI(
    title="Compass Master API",
    description="API for Compass Master application",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


register_tortoise(
    app,
    db_url="sqlite://db.sqlite3",
    modules={"models": ["database.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8000,
        log_level=env["LOG_LEVEL"].lower(),
        reload=False,
    )
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.env import env

app = FastAPI(
    title="Proposal Craft API",
    description="API for Proposal Craft application",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=env["HOST"],
        port=int(env["PORT"]),
        log_level=env["LOG_LEVEL"].lower(),
        reload=True,
    )
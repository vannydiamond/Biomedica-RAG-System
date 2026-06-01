 # FastAPI entry point

from fastapi import FastAPI
from .routers import qa_router

app = FastAPI()

app.include_router(qa_router.router)
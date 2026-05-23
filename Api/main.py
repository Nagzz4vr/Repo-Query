from fastapi import FastAPI

from Api.routes.ingest import router as ingest_router
from Api.routes.query import router as query_router

app = FastAPI(title="RAG Backend")

app.include_router(ingest_router)
app.include_router(query_router)
@app.get("/")
def root():
    return {"status": "ok"}


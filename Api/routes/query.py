
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from Api.dependencies import query_engine
from Api.schemas.query import QueryRequest

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/")
async def query(request: QueryRequest):

    try:

        result = await query_engine.query(
            request.question
        )

        return result.model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    
@router.post("/stream")
async def stream_query(request: QueryRequest):

    async def token_generator():

        async for token in query_engine.stream_query(
            request.question
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )

@router.post("/reset")
async def reset_memory():

    query_engine.reset_memory()

    return {
        "status": "memory_reset"
    }

@router.get("/debug/chunks")
async def debug_chunks(limit: int = 20, offset: int = 0):
    store = query_engine.vector_store
    records = list(store.records.items())
    page = records[offset : offset + limit]
    return {
        "total": store._next_idx,
        "offset": offset,
        "limit": limit,
        "chunks": [
            {
                "idx": idx,
                "id": r.id,
                "source": r.metadata.get("source", ""),
                "path": r.metadata.get("path", ""),
                "type": r.metadata.get("type", ""),
                "symbol": r.metadata.get("symbol_path", ""),
                "preview": r.document[:300],
            }
            for idx, r in page
        ],
    }
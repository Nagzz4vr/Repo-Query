
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
from fastapi import APIRouter, HTTPException

from Api.dependencies import write_pipeline
from Api.schemas.ingest import IngestRequest

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/")
async def ingest(request: IngestRequest):

    try:

        result = await write_pipeline.run(
            github_repo=request.github_repo,
            github_token=request.github_token,
            local_path=request.local_path,
            pdf_paths=request.pdf_paths,
            urls=request.urls,
        )

        return {
            "status": "success",
            "summary": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
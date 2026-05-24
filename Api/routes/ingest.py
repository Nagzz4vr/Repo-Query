from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from uuid import uuid4

from Api.dependencies import write_pipeline
from Api.schemas.ingest import IngestRequest

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_jobs: dict[str, dict] = {}

@router.post("/")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    _jobs[job_id] = {"status": "running", "summary": None, "error": None}

    async def _run():
        try:
            result = await write_pipeline.run(
                github_repo=request.github_repo,
                github_token=request.github_token,
                local_path=request.local_path,
                pdf_paths=request.pdf_paths,
                urls=request.urls,
            )
            _jobs[job_id] = {"status": "done", "summary": result, "error": None}
        except Exception as e:
            _jobs[job_id] = {"status": "failed", "summary": None, "error": str(e)}

    background_tasks.add_task(_run)

    return {"job_id": job_id, "status": "running"}


@router.get("/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
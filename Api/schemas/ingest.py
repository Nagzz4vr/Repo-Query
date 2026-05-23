# schemas/ingest.py

from pydantic import BaseModel
from typing import Optional


class IngestRequest(BaseModel):
    github_repo: Optional[str] = None
    github_token: Optional[str] = None
    local_path: Optional[str] = None
    pdf_paths: Optional[list[str]] = None
    urls: Optional[list[str]] = None
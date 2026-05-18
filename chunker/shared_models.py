
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class ChunkMethod(str, Enum):
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    TOKEN = "token"
    AST = "ast"   

class Chunk(BaseModel):
    chunk_index: int
    page_number: int                 
    text: str
    char_start: Optional[int] = None  
    method: ChunkMethod

    chunk_type: Optional[str] = None   # "function" | "class" | "method" | "file"
    chunk_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    path: Optional[str] = None
    language: Optional[str] = None


class ChunkedDocument(BaseModel):
    filepath: str
    total_pages: int
    chunk_method_used: ChunkMethod   
    total_chunks: int
    chunks: list[Chunk]
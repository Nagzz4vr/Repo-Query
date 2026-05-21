
from enum import Enum
from typing import Optional
from typing import Any, Optional, Dict,List,NamedTuple
from pydantic import BaseModel, Field

class ChunkMethod(str, Enum):
    AST            = "ast"
    SEMANTIC       = "semantic"
    RECURSIVE      = "recursive"
    TOKEN          = "token"
    FALLBACK_TEXT  = "fallback_text"

class SymbolType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    PROPERTY = "property"
    FILE = "file"


class RelationType(str, Enum):
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    DECORATES = "decorates"


class SymbolEdge(BaseModel):
    source_symbol_path: str 
    target_symbol_path: str 
    relation: RelationType


class SymbolIR(BaseModel):
    """Normalized, language-agnostic intermediate representation of an object."""
    symbol_path: str  
    name: str
    type: SymbolType
    parent_path: Optional[str] = None
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    code_segment: str
    signature: str     
    ast_hash: str      


class Chunk(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    text: str         
    raw_code: str      
    method: ChunkMethod
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    filepath: str
    chunk_method_used: ChunkMethod
    total_chunks: int
    chunks: List[Chunk]
    graph_edges: List[SymbolEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)



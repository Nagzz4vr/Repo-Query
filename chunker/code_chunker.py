from __future__ import annotations

import ast as _ast
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import NamedTuple
from Chunker.shared_models import Chunk, ChunkMethod, ChunkedDocument

logger = logging.getLogger("Code_Chunker")

class SourceFile(NamedTuple):
    filepath: str
    source_code: str
    language: str


class LanguageChunker(ABC):
    """Honest interface exposing language-agnostic code chunking pipelines."""
    @abstractmethod
    def chunk(self, source_file: SourceFile) -> ChunkedDocument:
        pass


class PythonASTChunker(LanguageChunker):
    """Encapsulates Python-specific parsing, relationship binding, and processing logic."""
    
    def chunk(self, source_file: SourceFile) -> ChunkedDocument:
        if source_file.language.lower() != "python":
            return self._fallback_to_text(source_file, reason="Unsupported language")

        try:
            tree = _ast.parse(source_file.source_code)
        except SyntaxError as e:
            logger.warning(f"AST parse failed for {source_file.filepath}: {e}")
            return self._fallback_to_text(source_file, reason=f"SyntaxError: {e}")

        lines = source_file.source_code.splitlines()
        processed_chunks: list[Chunk] = []
        
        # 1. Isolate imports at file scope rather than embedding them everywhere
        file_imports = self._extract_import_metadata(tree)

        # 2. Extract symbols cleanly, tracking parent-child hierarchies explicitly
        for node in tree.body:
            if isinstance(node, _ast.ClassDef):
                processed_chunks.extend(self._process_class(node, lines, source_file))
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                processed_chunks.append(self._build_chunk(node, lines, source_file, symbol_type="function"))

        if not processed_chunks:
            return self._fallback_to_text(source_file, reason="No class or function symbols discovered")

        return ChunkedDocument(
            filepath=source_file.filepath,
            chunk_method_used=ChunkMethod.AST,
            total_chunks=len(processed_chunks),
            chunks=processed_chunks,
            metadata={"global_imports": file_imports} 
        )

    def _process_class(self, class_node: _ast.ClassDef, lines: list[str], source_file: SourceFile) -> list[Chunk]:
        chunks = []
        # Parent Class Chunk (Stores the shell, attributes, and docstring safely)
        parent_chunk = self._build_chunk(class_node, lines, source_file, symbol_type="class")
        chunks.append(parent_chunk)
        
        for child in class_node.body:
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                # Child Method Chunk mapped directly via explicit parent relational reference
                child_chunk = self._build_chunk(
                    node=child, 
                    lines=lines, 
                    source_file=source_file, 
                    symbol_type="method", 
                    parent_id=parent_chunk.chunk_id
                )
                chunks.append(child_chunk)
        return chunks

    def _build_chunk(self, node: _ast.AST, lines: list[str], source_file: SourceFile, symbol_type: str, parent_id: str | None = None) -> Chunk:
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        
        # Defensively handle edge case where slice logic bounds overlap
        code_segment = "\n".join(lines[start_line - 1 : end_line])
        
        # Content addressed identity stabilization
        chunk_id = hashlib.blake2b(f"{source_file.filepath}:{start_line}:{code_segment}".encode("utf-8"), digest_size=16).hexdigest()

        return Chunk(
            chunk_id=chunk_id,
            parent_chunk_id=parent_id, # Clear relational line of sight
            text=code_segment,
            method=ChunkMethod.AST,
            chunk_type=symbol_type,
            chunk_name=getattr(node, "name", "anonymous"),
            start_line=start_line,
            end_line=end_line,
            docstring=_ast.get_docstring(node),
            path=source_file.filepath,
            language=source_file.language,
            dependencies=self._extract_node_dependencies(node) # Graph awareness
        )

    def _fallback_to_text(self, source_file: SourceFile, reason: str) -> ChunkedDocument:
        lines = source_file.source_code.splitlines()
        fallback_id = hashlib.blake2b(f"{source_file.filepath}:fallback".encode("utf-8"), digest_size=16).hexdigest()
        
        chunk = Chunk(
            chunk_id=fallback_id,
            text=source_file.source_code,
            method=ChunkMethod.FALLBACK_TEXT, # Observability problem fixed
            chunk_type="file",
            chunk_name=source_file.filepath,
            start_line=1,
            end_line=len(lines),
            path=source_file.filepath,
            language=source_file.language
        )
        return ChunkedDocument(
            filepath=source_file.filepath,
            chunk_method_used=ChunkMethod.FALLBACK_TEXT,
            total_chunks=1,
            chunks=[chunk],
            metadata={"fallback_reason": reason}
        )

    def _extract_import_metadata(self, tree: _ast.Module) -> list[str]:
        # Keeps import strings accessible at document envelope layer
        imports = []
        for node in tree.body:
            if isinstance(node, _ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, _ast.ImportFrom):
                imports.append(f"{node.module or ''}.{', '.join(a.name for a in node.names)}")
        return imports

    def _extract_node_dependencies(self, node: _ast.AST) -> list[str]:
        # Base symbol graph collection pass
        deps = []
        if isinstance(node, _ast.ClassDef):
            for base in node.bases:
                if isinstance(base, _ast.Name):
                    deps.append(base.id)
        # Scan internal nodes for call chains, decorators, etc. if required
        return deps

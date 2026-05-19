from __future__ import annotations

import ast as _ast
import hashlib
import logging
from typing import Any
from Chunker.shared_models import Chunk, ChunkMethod, ChunkedDocument 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("Code_Chunker")


class ASTChunker:
    def chunk(self, file_dict: dict[str, Any]) -> ChunkedDocument:
        language = file_dict["language"]

        if language == "python":
            return self._chunk_python(file_dict)

        return self._single_file_document(file_dict)
    
    def _chunk_python(self, file_dict: dict) -> ChunkedDocument:
        source = file_dict["source_code"]
        path = file_dict["path"]

        try:
            tree = _ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"AST parse failed for {path}: {e}")
            return self._single_file_document(file_dict)
        
        lines = source.splitlines()
        raw_chunks : list[dict] = []
        imports = self._extract_imports(tree, lines)

        for node in tree.body:
            if isinstance(node, _ast.ClassDef):
                raw_chunks.extend(
                    self._chunk_class(node, lines, imports, file_dict)
                )

            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                raw_chunks.append(
                    self._build_symbol_chunk(
                        node=node,
                        lines=lines,
                        imports=imports,
                        meta=file_dict,
                        symbol_type="function",
                        parent=None,
                    )
                )

        if not raw_chunks:
            return self._single_file_document(file_dict)
        
        chunks = [self._to_chunk_model(i, raw) for i, raw in enumerate(raw_chunks)]

        return  ChunkedDocument(
            filepath=file_dict["path"],
            total_pages=source.count("\n") + 1,   # lines ≈ "pages" for code
            chunk_method_used=ChunkMethod.AST,
            total_chunks=len(chunks),
            chunks=chunks,
        )
    def _chunk_class(self, class_node, lines, imports, meta) -> list[dict]:
        chunks = [self._build_symbol_chunk(class_node, lines, imports, meta, "class", None)]
        for child in class_node.body:
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                chunks.append(
                    self._build_symbol_chunk(child, lines, imports, meta, "method", class_node.name)
                )
        return chunks

    
    def _build_symbol_chunk(self, node, lines, imports, meta, symbol_type, parent) -> dict:
        end_line = getattr(node, "end_lineno", node.lineno)
        code = "\n".join(lines[node.lineno - 1: end_line])
        full_code = f"{imports}\n\n{code}".strip()

        return {
            "path": meta["path"],
            "language": meta["language"],
            "chunk_type": symbol_type,
            "chunk_name": node.name,
            "parent_symbol": parent,
            "start_line": node.lineno,
            "end_line": end_line,
            "docstring": _ast.get_docstring(node),
            "chunk_code": full_code,
        }

    def _to_chunk_model(self, index: int, raw: dict) -> Chunk:
        return Chunk(
            chunk_index=index,
            page_number=raw["start_line"],   
            text=raw["chunk_code"],
            char_start=None,
            method=ChunkMethod.AST,
            #code_chunker_specific
            chunk_type=raw["chunk_type"],
            chunk_name=raw["chunk_name"],
            parent_symbol=raw["parent_symbol"],
            start_line=raw["start_line"],
            end_line=raw["end_line"],
            docstring=raw["docstring"],
            path=raw["path"],
            language=raw["language"],
        )

    def _single_file_document(self, file_dict: dict) -> ChunkedDocument:
        source = file_dict["source_code"]
        total_lines = source.count("\n") + 1
        chunk = Chunk(
            chunk_index=0,
            page_number=1,
            text=source,
            char_start=0,
            method=ChunkMethod.AST,
            chunk_type="file",
            chunk_name=file_dict["path"],
            start_line=1,
            end_line=total_lines,
            path=file_dict["path"],
            language=file_dict["language"],
        )
        return ChunkedDocument(
            filepath=file_dict["path"],
            total_pages=total_lines,
            chunk_method_used=ChunkMethod.AST,
            total_chunks=1,
            chunks=[chunk],
        )

    def _extract_imports(self, tree, lines) -> str:
        imports = []
        for node in tree.body:
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                end_line = getattr(node, "end_lineno", node.lineno)
                imports.extend(lines[node.lineno - 1: end_line])
        return "\n".join(imports)

    def _hash_chunk(self, path: str, symbol: str, line: int) -> str:
        return hashlib.md5(f"{path}:{symbol}:{line}".encode()).hexdigest()
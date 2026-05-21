from __future__ import annotations

import ast as _ast
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import NamedTuple
from Chunker.shared_models import *
from typing import NamedTuple, Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
import time

logger = logging.getLogger("Code_Chunker")

class SourceFile(NamedTuple):
    filepath: str
    source_code: str
    language: str

class ParserResult(BaseModel):
    success: bool
    symbols: List[SymbolIR] = Field(default_factory=list)
    edges: List[SymbolEdge] = Field(default_factory=list)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class LanguageChunker(ABC):
    """Honest interface exposing language-agnostic code chunking pipelines."""
    @abstractmethod
    def chunk(self, source_file: SourceFile) -> ChunkedDocument:
        pass

class PythonASTParser(LanguageChunker):
    """Parses source files into structural symbols and walks dependencies."""
    
    def parse(self, source_file: SourceFile) -> ParserResult:
        start_time = time.perf_counter()
        
        try:
            tree = _ast.parse(source_file.source_code)
        except SyntaxError as e:
            logger.warning(f"AST parse failed for {source_file.filepath}: {e}")
            return ParserResult(success=False, error_message=f"SyntaxError: {e}")
            
        lines = source_file.source_code.splitlines()
        symbols: List[SymbolIR] = []
        edges: List[SymbolEdge] = []
        

        module_path = source_file.filepath.replace("/", ".").removesuffix(".py").strip(".")

        for node in tree.body:
            if isinstance(node, _ast.ClassDef):
                class_symbols, class_edges = self._process_class_node(node, lines, module_path)
                symbols.extend(class_symbols)
                edges.extend(class_edges)
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                symbols.append(self._build_symbol_ir(node, lines, module_path, SymbolType.FUNCTION))
                edges.extend(self._extract_node_dependencies(node, f"{module_path}.{node.name}"))

        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return ParserResult(
            success=True,
            symbols=symbols,
            edges=edges,
            metrics={"parse_time_ms": duration_ms}
        )
    def _process_class_node(self, class_node: _ast.ClassDef, lines: list[str], current_ns: str) -> tuple[list[SymbolIR], list[SymbolEdge]]:
        symbols = []
        edges = []
        class_path = f"{current_ns}.{class_node.name}"
        
        # Build the wrapper class container shell
        class_symbol = self._build_symbol_ir(class_node, lines, current_ns, SymbolType.CLASS)
        symbols.append(class_symbol)
        
        # Map out historical inheritance relationship properties 
        for base in class_node.bases:
            if isinstance(base, _ast.Name):
                edges.append(SymbolEdge(
                    source_symbol_path=class_path,
                    target_symbol_path=base.id, # Base string target reference name mapping tracker
                    relation=RelationType.INHERITS
                ))

        # Dig straight down into internal functional implementations
        for child in class_node.body:
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                stype = SymbolType.METHOD
                # Check for common decorators to give accurate classification
                decorator_names = [d.id for d in child.decorator_list if isinstance(d, _ast.Name)]
                if "staticmethod" in decorator_names:
                    stype = SymbolType.STATIC_METHOD
                elif "classmethod" in decorator_names:
                    stype = SymbolType.CLASS_METHOD
                
                method_symbol = self._build_symbol_ir(child, lines, class_path, stype, parent_path=class_path)
                symbols.append(method_symbol)
                
                # Link internal sub-method dependencies cleanly
                edges.extend(self._extract_node_dependencies(child, method_symbol.symbol_path))
                
        return symbols, edges

    def _build_symbol_ir(self, node: _ast.AST, lines: list[str], namespace: str, stype: SymbolType, parent_path: str | None = None) -> SymbolIR:
        name = getattr(node, "name", "anonymous")
        symbol_path = f"{namespace}.{name}"
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        
        code_segment = "\n".join(lines[start_line - 1 : end_line])
        
        # Isolate semantic execution footprint ignoring comments & formatting
        cloned = _ast.fix_missing_locations(node)
        ast_dump = _ast.dump(cloned, annotate_fields=False, include_attributes=False)
        ast_hash = hashlib.blake2b(ast_dump.encode("utf-8"), digest_size=16).hexdigest()

        # Isolate pure execution functional declaration templates 
        signature = f"def {name}"
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            args = _ast.unparse(node.args) if hasattr(_ast, "unparse") else "..."
            signature = f"{'async ' if isinstance(node, _ast.AsyncFunctionDef) else ''}def {name}({args})"
        elif isinstance(node, _ast.ClassDef):
            bases = ", ".join([_ast.unparse(b) for b in node.bases]) if hasattr(_ast, "unparse") else ""
            signature = f"class {name}({bases})"

        return SymbolIR(
            symbol_path=symbol_path,
            name=name,
            type=stype,
            parent_path=parent_path,
            start_line=start_line,
            end_line=end_line,
            docstring=_ast.get_docstring(node),
            code_segment=code_segment,
            signature=signature,
            ast_hash=ast_hash
        )

    def _extract_node_dependencies(self, node: _ast.AST, source_path: str) -> list[SymbolEdge]:
        edges = []
        for child in _ast.walk(node):
            if isinstance(child, _ast.Call) and isinstance(child.func, _ast.Name):
                edges.append(SymbolEdge(
                    source_symbol_path=source_path,
                    target_symbol_path=child.func.id,
                    relation=RelationType.CALLS
                ))
            elif isinstance(child, _ast.Attribute) and isinstance(child.value, _ast.Name):
                edges.append(SymbolEdge(
                    source_symbol_path=source_path,
                    target_symbol_path=f"{child.value.id}.{child.attr}",
                    relation=RelationType.CALLS
                ))
        return edges
    

class ChunkPolicyEngine:
    """Consolidates or filters nodes down dynamically based on user rules."""
    def __init__(self, min_lines: int = 4, merge_small_methods: bool = True):
        self.min_lines = min_lines
        self.merge_small_methods = merge_small_methods

    def apply_policies(self, symbols: List[SymbolIR]) -> List[SymbolIR]:
        optimized_symbols = []
        for symbol in symbols:
            line_count = (symbol.end_line - symbol.start_line) + 1
            # Filter trivial helper targets to keep retrieval collections uncluttered
            if line_count < self.min_lines and symbol.type == SymbolType.METHOD and self.merge_small_methods:
                continue
            optimized_symbols.append(symbol)
        return optimized_symbols


class ChunkProjectionFactory:
    """Transforms raw AST objects into data configurations optimized for high-recall vector search."""
    @staticmethod
    def create_lexical_search_text(symbol: SymbolIR) -> str:
        return f"""Symbol: {symbol.name}
Type: {symbol.type.value}
Path: {symbol.symbol_path}
Signature: {symbol.signature}
Docstring: {symbol.docstring or 'None provided'}
Code:
{symbol.code_segment}"""

    @staticmethod
    def materialize(symbol: SymbolIR, parser_metrics: Dict[str, Any]) -> Chunk:
        retrieval_text = ChunkProjectionFactory.create_lexical_search_text(symbol)
        
        # Build deterministic identity configurations based on content footprint tracking
        chunk_id = hashlib.blake2b(f"{symbol.symbol_path}:{symbol.ast_hash}".encode("utf-8"), digest_size=16).hexdigest()
        
        parent_chunk_id = None
        if symbol.parent_path:
            parent_chunk_id = hashlib.blake2b(f"{symbol.parent_path}:parent".encode("utf-8"), digest_size=16).hexdigest()

        return Chunk(
            chunk_id=chunk_id,
            parent_chunk_id=parent_chunk_id,
            text=retrieval_text,
            raw_code=symbol.code_segment,
            method=ChunkMethod.AST,
            metadata={
                "symbol_path": symbol.symbol_path,
                "name": symbol.name,
                "type": symbol.type.value,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "ast_hash": symbol.ast_hash,
                "parse_time_ms": parser_metrics.get("parse_time_ms", 0.0)
            }
        )
    
class FallbackTextChunker:
    """Fallback handler that isolates processing failures cleanly away from primary pipelines."""
    def chunk(self, source_file: SourceFile, reason: str) -> ChunkedDocument:
        lines = source_file.source_code.splitlines()
        fallback_id = hashlib.blake2b(f"{source_file.filepath}:fallback".encode("utf-8"), digest_size=16).hexdigest()
        
        chunk = Chunk(
            chunk_id=fallback_id,
            text=f"File: {source_file.filepath}\nLanguage: {source_file.language}\nRaw Text Context:\n{source_file.source_code}",
            raw_code=source_file.source_code,
            method=ChunkMethod.FALLBACK_TEXT,
            metadata={
                "type": SymbolType.FILE.value,
                "start_line": 1,
                "end_line": len(lines),
                "fallback_reason": reason
            }
        )
        return ChunkedDocument(
            filepath=source_file.filepath,
            chunk_method_used=ChunkMethod.FALLBACK_TEXT,
            total_chunks=1,
            chunks=[chunk],
            graph_edges=[],
            metadata={"fallback_reason": reason}
        )


class CodeIntelligenceOrchestrator:
    """The master pipeline container execution router workflow."""
    def __init__(self, parser_registry: Dict[str, LanguageParser], policy_engine: ChunkPolicyEngine):
        self.parser_registry = parser_registry
        self.policy_engine = policy_engine
        self.fallback_handler = FallbackTextChunker()

    def process_file(self, source_file: SourceFile) -> ChunkedDocument:
        parser = self.parser_registry.get(source_file.language.lower())
        
        # Explicit policy decision instead of an inline exception catch loop
        if not parser:
            return self.fallback_handler.chunk(source_file, reason=f"Unsupported language: '{source_file.language}'")
            
        parse_result = parser.parse(source_file)
        if not parse_result.success:
            return self.fallback_handler.chunk(source_file, reason=parse_result.error_message or "Unknown parsing failure")
            
        if not parse_result.symbols:
            return self.fallback_handler.chunk(source_file, reason="No symbols extracted from valid source module layout definitions.")

        # Filter and optimize chunks using policies
        filtered_symbols = self.policy_engine.apply_policies(parse_result.symbols)
        
        final_chunks = [
            ChunkProjectionFactory.materialize(symbol, parse_result.metrics) 
            for symbol in filtered_symbols
        ]
        
        return ChunkedDocument(
            filepath=source_file.filepath,
            chunk_method_used=ChunkMethod.AST,
            total_chunks=len(final_chunks),
            chunks=final_chunks,
            graph_edges=parse_result.edges,
            metadata={"parse_metrics": parse_result.metrics}
        )
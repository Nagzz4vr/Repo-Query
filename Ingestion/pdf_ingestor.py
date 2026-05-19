from __future__ import annotations
 
import io
import os
from pathlib import Path
from typing import Optional

import pymupdf
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator


#input

class IngestorConfig(BaseModel):
    max_file_size_mb: float = Field(default=50.0, gt=0, description="Maximum PDF size in MB")
    max_pages: int = Field(default=500, gt=0, description="Maximum number of pages allowed")
    scanned_text_threshold: int = Field(
        default=40,
        ge=0,
        description="Pages with fewer native characters than this are treated as scanned",
    )
    ocr_lang: str = Field(default="eng", description="Tesseract language string")
    extract_tables: bool = Field(default=True, description="Run table extraction where detected")

    @field_validator("ocr_lang")
    @classmethod
    def lang_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ocr_lang cannot be empty")
        return v.strip()
    

#output
class BoundingBox(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]  
    font: str
    size: float

class LineModel(BaseModel):
    spans: list[BoundingBox]


class BlockModel(BaseModel):
    block_no: int
    block_type: int   
    bbox: tuple[float, float, float, float]
    lines: list[LineModel] = Field(default_factory=list)

class PageLayout(BaseModel):
    page_number: int                     
    width: float
    height: float
    blocks: list[BlockModel]
    native_char_count: int                
    has_images: bool
    likely_scanned: bool                  
    likely_has_tables: bool  

class LayoutMap(BaseModel):

    total_pages: int
    pages: list[PageLayout]

#result
class PageTextResult(BaseModel):
    page_number: int
    text: str
    extraction_method: str

class PageTableResult(BaseModel):
    page_number: int
    tables: list[list[list[str | None]]]

class PageResult(BaseModel):
    page_number: int
    text: PageTextResult
    tables: Optional[PageTableResult] = None

class ExtractionResult(BaseModel):

    filepath: str
    total_pages: int
    file_size_mb: float
    config: IngestorConfig
    layout_map: LayoutMap
    pages: list[PageResult]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text.text for p in self.pages)

    @property
    def all_tables(self) -> list[PageTableResult]:
        return [p.tables for p in self.pages if p.tables]



class SecurityError(Exception):
    """Raised when path validation fails for security reasons"""
    pass

class PdfIngestor:
    def __init__(self,filepath: str, base_dir: str = None):
        if base_dir is None:
            base_dir=os.getcwd()

        self.base_dir=Path(base_dir).resolve()
        self.filepath=self._validate_safe_path(filepath)
        self.config=IngestorConfig()
        self._check_format()
        self._validate_exists()

    def _validate_safe_path(self, user_path: str) -> Path:
        """Validate user_path is within base_dir"""
        if not user_path or not user_path.strip():
            raise ValueError("Path cannot be empty or whitespace")
        
        target = (self.base_dir / user_path).resolve()
        
        if target == self.base_dir:
            raise ValueError(f"Path resolves to base directory itself: {target}")
        
        if self.base_dir not in target.parents:
            raise SecurityError(
                f"Path traversal detected: '{user_path}' resolves to '{target}', "
                f"which is outside base directory '{self.base_dir}'"
            )
        
        return target
    
    def _validate_exists(self):
        """Check if file exists"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
    
    def _check_format(self):
        if self.filepath.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported file format: {self.filepath.suffix}. "
            )
    
    def validate(self) -> tuple[float, int]:

        errors: list[str] = []
        size_bytes = self.filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            errors.append(
                f"File size {size_mb:.2f} MB exceeds limit of {self.config.max_file_size_mb} MB"
            )

        page_count = 0
        doc = pymupdf.open(str(self.filepath))
        try:
            page_count = len(doc)
        finally:
            doc.close()
        if page_count > self.config.max_pages:
            errors.append(
                f"Page count {page_count} exceeds limit of {self.config.max_pages}"
            )
        if errors:
            raise ValueError("Validation failed:\n  " + "\n  ".join(errors))
        return size_mb, page_count
    
    def build_layout_map(self, doc: pymupdf.Document) -> LayoutMap:

        pages:list[PageLayout]=[]
        for page_num,page in enumerate(doc):
            data=page.get_text("dict")
            page_w=page.rect.width
            page_h=page.rect.height

            blocks:list[BlockModel]=[]
            native_chars=0
            has_images=False

            for block in data.get("blocks",[]):
                btype=block.get("type",0)
                bbox=block.get("bbox", (0,0,0,0))
                #image block
                if btype==1:
                    has_images=True
                    blocks.append(
                        BlockModel(
                            block_no=block.get("number", 0),
                            block_type=btype,
                            bbox=bbox,
                        )
                    )
                    continue
                #text block
                lines:list[LineModel]=[]
                for line in block.get("lines",[]):
                    spans: list[BoundingBox] = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        native_chars += len(text)
                        spans.append(
                            BoundingBox(
                                text=text,
                                bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                                font=span.get("font", ""),
                                size=span.get("size", 0.0),
                            )
                        )
                    lines.append(LineModel(spans=spans))
                blocks.append(
                    BlockModel(
                        block_no=block.get("number",0),
                        block_type=btype,
                        bbox=bbox,
                        lines=lines
                    )
                )
                likely_scanned = native_chars < self.config.scanned_text_threshold
                likely_has_tables = self._heuristic_table_detection(blocks, page_w)

            pages.append(
                    PageLayout(
                        page_number=page_num,
                        width=page_w,
                        height=page_h,
                        blocks=blocks,
                        native_char_count=native_chars,
                        has_images=has_images,
                        likely_scanned=likely_scanned,
                        likely_has_tables=likely_has_tables,
                    )
                )

        return LayoutMap(total_pages=len(pages), pages=pages)
    
    @staticmethod
    def _heuristic_table_detection(blocks: list[BlockModel], page_width: float) -> bool:
        text_blocks = [b for b in blocks if b.block_type == 0]
        if len(text_blocks) < 3:
            return False

        narrow_threshold = page_width * 0.40
        y_positions: dict[int, int] = {}
        for block in text_blocks:
            x0, y0, x1, y1 = block.bbox
            width = x1 - x0
            if width < narrow_threshold:
                bucket = int(y0 / 5) * 5   
                y_positions[bucket] = y_positions.get(bucket, 0) + 1
        return any(count >= 3 for count in y_positions.values())
    


    def _extract_native_text(self, page: pymupdf.Page) -> str:
        return page.get_text("text")

    def _extract_ocr(self, page: pymupdf.Page) -> str:
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for OCR. Install with: pip install pytesseract"
            ) from exc

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=self.config.ocr_lang)

    def _extract_tables(self, page: pymupdf.Page) -> list[list[list[str | None]]]:
        table_finder = page.find_tables()
        return [t.extract() for t in table_finder.tables]

    def extract(self)->ExtractionResult:
        file_size_mb, total_pages = self.validate()
        doc = pymupdf.open(str(self.filepath))
        try:
            layout_map=self.build_layout_map(doc)

            page_results:list[PageResult]=[]
            for layout in layout_map.pages:
                page=doc[layout.page_number]

                if layout.likely_scanned:
                    raw_text = self._extract_ocr(page)
                    method = "ocr"
                elif layout.native_char_count == 0:
                    raw_text = ""
                    method = "empty"
                else:
                    raw_text = self._extract_native_text(page)
                    method = "native"

                text_result = PageTextResult(
                    page_number=layout.page_number,
                    text=raw_text,
                    extraction_method=method,
                )
                table_result: Optional[PageTableResult] = None
                if self.config.extract_tables and layout.likely_has_tables:
                    raw_tables = self._extract_tables(page)
                    if raw_tables:
                        table_result = PageTableResult(
                            page_number=layout.page_number,
                            tables=raw_tables,
                        )

                page_results.append(
                    PageResult(
                        page_number=layout.page_number,
                        text=text_result,
                        tables=table_result,
                    )
                )

        finally:
            doc.close()

        return ExtractionResult(
            filepath=str(self.filepath),
            total_pages=total_pages,
            file_size_mb=round(file_size_mb, 3),
            config=self.config,
            layout_map=layout_map,
            pages=page_results,
        )

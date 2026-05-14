from pydantic import BaseModel, Field
from typing import Type, TypeVar, Optional, Any
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path

T = TypeVar('T', bound=BaseModel)

class ScrapedMetadata(BaseModel):
    """Metadata for chunking and downstream processing"""
    url: str
    title: str
    scraped_at: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    content_hash: str
    headers: dict[str, str] = Field(default_factory=dict)
    links_count: int
    images_count: int
    success: bool
    error_message: Optional[str] = None
    

    content_type: str 
    language: Optional[str] = None
    word_count: int
    has_structured_data: bool

class ExtractionResult(BaseModel):
    """Complete extraction result with metadata"""
    metadata: ScrapedMetadata
    extracted_data: dict[str, Any] 
    raw_markdown: str  
    compressed_markdown: str  
    chunks_metadata: Optional[dict] = None  



class UniversalScraper:

    
    def __init__(
        self,
        provider: str = "openai/gpt-4o-mini",
        api_token: Optional[str] = None,
        max_tokens: int = 4000,
        token_threshold: float = 0.5,
        verbose: bool = True
    ):
        self.provider = provider
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.token_threshold = token_threshold
        self.verbose = verbose
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _calculate_hash(self, text: str) -> str:
        """Generate content hash for deduplication"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    async def scrape_and_extract(
        self,
        url: str,
        schema: Type[T],
        instruction: str,
        content_type: str = "general",
        save_to_disk: bool = False,
        output_dir: str = "./scraped_data"
    ) -> ExtractionResult:
        """
        Scrape, compress, extract, and preserve metadata
        
        Args:
            url: Target URL
            schema: Pydantic model for extraction
            instruction: LLM extraction instruction
            content_type: Type of content for metadata
            save_to_disk: Whether to save results
            output_dir: Directory for saved results
        
        Returns:
            ExtractionResult with metadata and extracted data
        """
        
        async with AsyncWebCrawler(verbose=self.verbose) as crawler:
            try:
            
                if self.verbose:
                    print(f"\n Crawling: {url}")
                
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    word_count_threshold=10,
                    excluded_tags=['form', 'nav', 'footer', 'header'],
                    remove_overlay_elements=True
                )
                
                if not result.success:
                    raise Exception(f"Crawl failed: {result.error_message}")
                
            
                if self.verbose:
                    print(f" Compressing markdown...")
                
                original_markdown = result.markdown.raw_markdown
                compressed_markdown = result.markdown.fit_markdown(
                    max_tokens=self.max_tokens,
                    token_threshold=self.token_threshold
                )
                
                original_tokens = self._estimate_tokens(original_markdown)
                compressed_tokens = self._estimate_tokens(compressed_markdown)
                compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
                
                if self.verbose:
                    print(f"   Original: ~{original_tokens} tokens")
                    print(f"   Compressed: ~{compressed_tokens} tokens")
                    print(f"   Ratio: {compression_ratio:.2%}")
                
            
                if self.verbose:
                    print(f" Extracting structured data...")
                
                strategy = LLMExtractionStrategy(
                    provider=self.provider,
                    api_token=self.api_token,
                    schema=schema.model_json_schema(),
                    instruction=instruction,
                    verbose=self.verbose
                )
                
    
                extracted_json = await strategy.extract(
                    url=result.url,
                    html=compressed_markdown
                )
                
                # Validate with Pydantic
                extracted_data = schema.model_validate_json(extracted_json)
                
    
                metadata = ScrapedMetadata(
                    url=url,
                    title=result.metadata.get('title', 'Unknown'),
                    scraped_at=datetime.utcnow().isoformat(),
                    original_token_count=original_tokens,
                    compressed_token_count=compressed_tokens,
                    compression_ratio=compression_ratio,
                    content_hash=self._calculate_hash(original_markdown),
                    headers=dict(result.metadata.get('headers', {})),
                    links_count=len(result.links.get('internal', [])) + len(result.links.get('external', [])),
                    images_count=len(result.media.get('images', [])),
                    success=True,
                    error_message=None,
                    content_type=content_type,
                    language=result.metadata.get('language'),
                    word_count=len(original_markdown.split()),
                    has_structured_data=True
                )
                

                extraction_result = ExtractionResult(
                    metadata=metadata,
                    extracted_data=extracted_data.model_dump(),
                    raw_markdown=original_markdown,
                    compressed_markdown=compressed_markdown
                )

                if save_to_disk:
                    self._save_results(extraction_result, output_dir)
                
                if self.verbose:
                    print(f" Extraction complete!")
                
                return extraction_result
                
            except Exception as e:
                # Handle errors gracefully
                error_metadata = ScrapedMetadata(
                    url=url,
                    title="Error",
                    scraped_at=datetime.utcnow().isoformat(),
                    original_token_count=0,
                    compressed_token_count=0,
                    compression_ratio=0.0,
                    content_hash="",
                    links_count=0,
                    images_count=0,
                    success=False,
                    error_message=str(e),
                    content_type=content_type,
                    word_count=0,
                    has_structured_data=False
                )
                
                return ExtractionResult(
                    metadata=error_metadata,
                    extracted_data={},
                    raw_markdown="",
                    compressed_markdown=""
                )
    
    def _save_results(self, result: ExtractionResult, output_dir: str):
        """Save results to disk with organized structure"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create timestamp-based filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_url = result.metadata.url.replace("https://", "").replace("http://", "").replace("/", "_")[:50]
        base_filename = f"{timestamp}_{safe_url}"
        
        # Save metadata
        with open(f"{output_dir}/{base_filename}_metadata.json", "w") as f:
            json.dump(result.metadata.model_dump(), f, indent=2)
        
        # Save extracted data
        with open(f"{output_dir}/{base_filename}_extracted.json", "w") as f:
            json.dump(result.extracted_data, f, indent=2)
        
        # Save compressed markdown (for chunking)
        with open(f"{output_dir}/{base_filename}_compressed.md", "w") as f:
            f.write(result.compressed_markdown)
        
        # Save raw markdown (backup)
        with open(f"{output_dir}/{base_filename}_raw.md", "w") as f:
            f.write(result.raw_markdown)
        
        print(f" Saved to: {output_dir}/{base_filename}_*")

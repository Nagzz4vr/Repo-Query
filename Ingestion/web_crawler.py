from pydantic import BaseModel, Field, model_validator
from typing import  Generic, Optional, Type, TypeVar,Any
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)
import logging
from datetime import datetime, timezone
import hashlib 

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_TRANSIENT = (
    ConnectionError,
    TimeoutError,
    OSError,        
)

def _make_retry_decorator(max_attempts: int, wait_min: float, wait_max: float):
    """Factory so callers can tune retry behaviour per-instance."""
    return retry(
        retry=retry_if_exception_type(_TRANSIENT),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

class ScrapedMetadata(BaseModel):
    url: str
    title: str
    scraped_at: str         
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    content_hash: str       
    links_count: int
    images_count: int
    success: bool
    error_message: Optional[str] = None
    content_type: str
    language: Optional[str] = None
    word_count: int
    has_structured_data: bool
    headers: dict[str, str] = Field(default_factory=dict)
 
    @model_validator(mode="before")
    @classmethod
    def _coerce_headers(cls, values: dict) -> dict:
        raw = values.get("headers", {})
        values["headers"] = {str(k): str(v) for k, v in raw.items()}
        return values

class ExtractionResult(BaseModel, Generic[T]):

    metadata: ScrapedMetadata
    extracted_data: Optional[T]          # None on failure
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
        max_attempts: int = 3,
        wait_min: float = 2.0,
        wait_max: float = 30.0,
        verbose: bool = False,
    ):
        self.provider = provider
        self.api_token = (
            api_token
            or os.getenv("GROQ_API_KEY")
        )
        if not self.api_token:
            raise ValueError(
                "No API token provided. Pass api_token= or set "
                "OPENAI_API_KEY / ANTHROPIC_API_KEY."
            )

        self.max_tokens = max_tokens
        self.token_threshold = token_threshold
        self.max_attempts = max_attempts
        self.wait_min = wait_min
        self.wait_max = wait_max

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        self._strategy_cache: dict[tuple, LLMExtractionStrategy] = {}
    
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
            output_dir: str = "./scraped_data",
        ) -> ExtractionResult[T]:
        retry_fn = _make_retry_decorator(self.max_attempts, self.wait_min, self.wait_max)

        try:
            result = await retry_fn(self._scrape_and_extract_once)(
                url=url,
                schema=schema,
                instruction=instruction,
                content_type=content_type,
            )
        except RetryError as exc:

            logger.error("All %d attempts failed for %s: %s", self.max_attempts, url, exc)
            result = self._failure_result(url, content_type, str(exc))

        if save_to_disk:
            self._save_results(result, output_dir)
        return result
    
    async def _scrape_and_extract_once(
        self,
        url: str,
        schema: Type[T],
        instruction: str,
        content_type: str,
    ) -> ExtractionResult[T]:
        logger.debug("Crawling: %s", url)
        run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            excluded_tags=["form", "nav", "footer", "header"],
            remove_overlay_elements=True,
            fit_markdown=True,                 
            max_tokens=self.max_tokens,      
            token_threshold=self.token_threshold,
        )

        async with AsyncWebCrawler() as crawler:
                crawl_result = await crawler.arun(url=url, config=run_cfg)
        if not crawl_result.success:
            raise ConnectionError(
                f"Crawl failed for {url}: {crawl_result.error_message}"
            )
        raw_markdown: str = crawl_result.markdown.raw_markdown

        compressed_markdown: str = (
            crawl_result.markdown.fit_markdown or raw_markdown
        )

        original_tokens = self._estimate_tokens(raw_markdown)
        compressed_tokens = self._estimate_tokens(compressed_markdown)
        compression_ratio = (
            compressed_tokens / original_tokens if original_tokens else 0.0
        )
        logger.debug(
            "Tokens — original: %d  compressed: %d  ratio: %.1f%%",
            original_tokens,
            compressed_tokens,
            compression_ratio * 100,
        )

        logger.debug("Extracting structured data with %s", self.provider)
        strategy = self._get_strategy(schema, instruction)

        try:
            extracted_json: str = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: strategy.extract(
                    url=crawl_result.url,
                    html=compressed_markdown,
                ),
            )
        except Exception as exc:
            raise ConnectionError(f"LLM extraction failed: {exc}") from exc
        

        try:
            extracted_data: T = schema.model_validate_json(extracted_json)
        except Exception as exc:
            logger.warning("Schema validation failed: %s", exc)
            logger.debug("Raw LLM output: %s", extracted_json)
            return self._partial_result(
                url=url,
                content_type=content_type,
                raw_markdown=raw_markdown,
                compressed_markdown=compressed_markdown,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                crawl_result=crawl_result,
                error=f"Schema validation error: {exc}",
            )
        
        metadata = ScrapedMetadata(
            url=url,
            title=crawl_result.metadata.get("title", "Unknown"),
            scraped_at=datetime.now(timezone.utc).isoformat(),
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compression_ratio,
            content_hash=self._calculate_hash(raw_markdown),
            headers=dict(crawl_result.metadata.get("headers", {})),
            links_count=(
                len(crawl_result.links.get("internal", []))
                + len(crawl_result.links.get("external", []))
            ),
            images_count=len(crawl_result.media.get("images", [])),
            success=True,
            content_type=content_type,
            language=crawl_result.metadata.get("language"),
            word_count=len(raw_markdown.split()),
            has_structured_data=True,
        )

        logger.debug("Extraction complete for %s", url)
        return ExtractionResult(
            metadata=metadata,
            extracted_data=extracted_data,
            raw_markdown=raw_markdown,
            compressed_markdown=compressed_markdown,
        )

    def _get_strategy(self, schema: Type[T], instruction: str) -> LLMExtractionStrategy:
        key = (json.dumps(schema.model_json_schema(), sort_keys=True), instruction)
        if key not in self._strategy_cache:
            self._strategy_cache[key] = LLMExtractionStrategy(
                provider=self.provider,
                api_token=self.api_token,
                schema=schema.model_json_schema(),
                instruction=instruction,
            )
        return self._strategy_cache[key]
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(len(text) // 4, 0)
 
    @staticmethod
    def _calculate_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _failure_result(
        self, url: str, content_type: str, error: str
    ) -> ExtractionResult:
        return ExtractionResult(
            metadata=ScrapedMetadata(
                url=url,
                title="Error",
                scraped_at=datetime.now(timezone.utc).isoformat(),
                original_token_count=0,
                compressed_token_count=0,
                compression_ratio=0.0,
                content_hash="",
                links_count=0,
                images_count=0,
                success=False,
                error_message=error,
                content_type=content_type,
                word_count=0,
                has_structured_data=False,
            ),
            extracted_data=None,
            raw_markdown="",
            compressed_markdown="",
        )
    
    def _partial_result(
        self,
        url: str,
        content_type: str,
        raw_markdown: str,
        compressed_markdown: str,
        original_tokens: int,
        compressed_tokens: int,
        compression_ratio: float,
        crawl_result: Any,
        error: str,
    ) -> ExtractionResult:
        return ExtractionResult(
            metadata=ScrapedMetadata(
                url=url,
                title=crawl_result.metadata.get("title", "Unknown"),
                scraped_at=datetime.now(timezone.utc).isoformat(),
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                compression_ratio=compression_ratio,
                content_hash=self._sha256(raw_markdown),
                headers=dict(crawl_result.metadata.get("headers", {})),
                links_count=(
                    len(crawl_result.links.get("internal", []))
                    + len(crawl_result.links.get("external", []))
                ),
                images_count=len(crawl_result.media.get("images", [])),
                success=False,
                error_message=error,
                content_type=content_type,
                language=crawl_result.metadata.get("language"),
                word_count=len(raw_markdown.split()),
                has_structured_data=False,
            ),
            extracted_data=None,
            raw_markdown=raw_markdown,
            compressed_markdown=compressed_markdown,
        )
    def _save_results(self, result: ExtractionResult, output_dir: str):
        """Save results to disk with organized structure"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create timestamp-based filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_url = result.metadata.url.replace("https://", "").replace("http://", "").replace("/", "_")[:50]
        base_filename = f"{timestamp}_{safe_url}"
        

        with open(f"{output_dir}/{base_filename}_metadata.json", "w") as f:
            json.dump(result.metadata.model_dump(), f, indent=2)

        with open(f"{output_dir}/{base_filename}_extracted.json", "w") as f:
            json.dump(result.extracted_data.model_dump(), f, indent=2)
        

        with open(f"{output_dir}/{base_filename}_compressed.md", "w") as f:
            f.write(result.compressed_markdown)
        
        
        with open(f"{output_dir}/{base_filename}_raw.md", "w") as f:
            f.write(result.raw_markdown)
        
        print(f" Saved to: {output_dir}/{base_filename}_*")
# tests/test_models.py
import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from Pipelines.ingestion_pipeline import *

# =========================================================
# ENUM TESTS
# =========================================================

def test_content_format_values():
    assert ContentFormat.PLAIN_TEXT == "plain_text"
    assert ContentFormat.MARKDOWN == "markdown"
    assert ContentFormat.SOURCE_CODE == "source_code"


def test_source_type_values():
    assert SourceType.CODE == "code"
    assert SourceType.PDF == "pdf"
    assert SourceType.WEB == "web"


# =========================================================
# PROVENANCE TESTS
# =========================================================

def test_provenance_creation():
    provenance = Provenance(
        source_uri="file:///tmp/test.py",
        checksum="abc123"
    )

    assert provenance.source_uri == "file:///tmp/test.py"
    assert provenance.checksum == "abc123"
    assert provenance.version is None


def test_provenance_with_optional_fields():
    now = datetime.utcnow()

    provenance = Provenance(
        source_uri="https://github.com/test/repo",
        checksum="sha256",
        version="1.0.0",
        branch="main",
        commit_sha="deadbeef",
        page_number=5,
        section_title="Introduction",
        line_number=42,
        scraped_at=now,
        modified_at=now,
    )

    assert provenance.branch == "main"
    assert provenance.page_number == 5
    assert provenance.line_number == 42


def test_provenance_requires_source_uri():
    with pytest.raises(ValidationError):
        Provenance(checksum="abc")


def test_provenance_requires_checksum():
    with pytest.raises(ValidationError):
        Provenance(source_uri="file://test")


# =========================================================
# RAW DOCUMENT TESTS
# =========================================================

def test_raw_document_creation():
    provenance = Provenance(
        source_uri="file:///test.py",
        checksum="123"
    )

    doc = RawDocument(
        source_type=SourceType.CODE,
        source_uri="file:///test.py",
        content="print('hello')",
        content_format=ContentFormat.SOURCE_CODE,
        provenance=provenance,
    )

    assert doc.source_type == SourceType.CODE
    assert doc.content == "print('hello')"
    assert doc.metadata == {}
    assert doc.structured_data is None


def test_raw_document_generates_document_id():
    provenance = Provenance(
        source_uri="file:///test.py",
        checksum="123"
    )

    doc = RawDocument(
        source_type=SourceType.CODE,
        source_uri="file:///test.py",
        content="hello",
        content_format=ContentFormat.PLAIN_TEXT,
        provenance=provenance,
    )

    assert doc.document_id.startswith("doc_")
    assert len(doc.document_id) > 10


def test_raw_document_created_at_auto_generated():
    provenance = Provenance(
        source_uri="file:///test.py",
        checksum="123"
    )

    doc = RawDocument(
        source_type=SourceType.CODE,
        source_uri="file:///test.py",
        content="hello",
        content_format=ContentFormat.PLAIN_TEXT,
        provenance=provenance,
    )

    assert isinstance(doc.created_at, datetime)


def test_raw_document_with_metadata():
    provenance = Provenance(
        source_uri="file:///test.py",
        checksum="123"
    )

    metadata = {
        "language": "python",
        "framework": "fastapi"
    }

    doc = RawDocument(
        source_type=SourceType.CODE,
        source_uri="file:///test.py",
        content="hello",
        content_format=ContentFormat.SOURCE_CODE,
        provenance=provenance,
        metadata=metadata,
    )

    assert doc.metadata["language"] == "python"


def test_raw_document_with_structured_data():
    provenance = Provenance(
        source_uri="https://example.com",
        checksum="123"
    )

    structured_data = {
        "title": "Example",
        "author": "Test"
    }

    doc = RawDocument(
        source_type=SourceType.WEB,
        source_uri="https://example.com",
        content="content",
        content_format=ContentFormat.HTML,
        provenance=provenance,
        structured_data=structured_data,
    )

    assert doc.structured_data["title"] == "Example"


def test_raw_document_missing_required_fields():
    provenance = Provenance(
        source_uri="file:///test.py",
        checksum="123"
    )

    with pytest.raises(ValidationError):
        RawDocument(
            source_type=SourceType.CODE,
            provenance=provenance,
        )


# =========================================================
# INGESTION REQUEST TESTS
# =========================================================

def test_code_ingestion_request_defaults():
    request = CodeIngestionRequest()

    assert request.source_type == SourceType.CODE
    assert request.github_branch == "main"
    assert request.max_file_size == 500_000


def test_code_ingestion_request_local_path():
    request = CodeIngestionRequest(
        local_path="/tmp/project"
    )

    assert request.local_path == "/tmp/project"


def test_code_ingestion_request_github_repo():
    request = CodeIngestionRequest(
        github_repo="user/repo",
        github_token="secret"
    )

    assert request.github_repo == "user/repo"
    assert request.github_token == "secret"


def test_pdf_ingestion_request_creation():
    request = PdfIngestionRequest(
        filepath="sample.pdf"
    )

    assert request.source_type == SourceType.PDF
    assert request.filepath == "sample.pdf"
    assert request.max_pages == 500
    assert request.extract_tables is True


def test_pdf_ingestion_request_custom_values():
    request = PdfIngestionRequest(
        filepath="sample.pdf",
        max_pages=100,
        extract_tables=False,
        ocr_lang="jpn"
    )

    assert request.max_pages == 100
    assert request.extract_tables is False
    assert request.ocr_lang == "jpn"


def test_pdf_ingestion_request_requires_filepath():
    with pytest.raises(ValidationError):
        PdfIngestionRequest()


def test_web_ingestion_request_creation():
    request = WebIngestionRequest(
        url="https://example.com"
    )

    assert request.source_type == SourceType.WEB
    assert request.url == "https://example.com"
    assert request.provider == "openai/gpt-4o-mini"


def test_web_ingestion_request_with_schema():
    schema = {
        "title": "string",
        "price": "float"
    }

    request = WebIngestionRequest(
        url="https://example.com",
        schema_definition=schema,
        instruction="Extract products"
    )

    assert request.schema_definition == schema
    assert request.instruction == "Extract products"


def test_web_ingestion_request_requires_url():
    with pytest.raises(ValidationError):
        WebIngestionRequest()


# =========================================================
# BASE INGESTOR TESTS
# =========================================================

class DummyIngestor(BaseIngestor):

    async def ingest(self, request):
        provenance = Provenance(
            source_uri="dummy://test",
            checksum="123"
        )

        yield RawDocument(
            source_type=SourceType.CODE,
            source_uri="dummy://test",
            content="hello",
            content_format=ContentFormat.PLAIN_TEXT,
            provenance=provenance,
        )


@pytest.mark.asyncio
async def test_base_ingestor_streaming():
    ingestor = DummyIngestor()

    docs = []

    async for doc in ingestor.ingest(None):
        docs.append(doc)

    assert len(docs) == 1
    assert docs[0].content == "hello"


def test_base_ingestor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseIngestor()
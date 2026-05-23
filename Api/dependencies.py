
from uuid import uuid4

from Embedder.embedder import Embedder
from LLM.llm_client import LiteLLMClient
from Pipelines.write_pipeline import WritePipeline
from Query.query_engine import QueryEngine


embedder = Embedder(
    model_name="BAAI/bge-m3"
)

llm_client = LiteLLMClient(
    model_pool=[
        "groq/llama-3.3-70b-versatile",
    ],
    request_id=str(uuid4()),
    session_id="global_session",
)

write_pipeline = WritePipeline()

query_engine = QueryEngine(
    client=llm_client,
    embedder=embedder,
    vector_store=write_pipeline.vector_store,
)
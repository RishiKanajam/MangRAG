from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    source: str = Field(..., description="PDF URL or local file path")


class IngestResponse(BaseModel):
    source: str
    chunks_stored: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    content: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[RetrievedChunk]


class StatsResponse(BaseModel):
    total_chunks: int
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    mongodb: str
    embeddings: str

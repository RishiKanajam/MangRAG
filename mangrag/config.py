from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    groq_api_key: str
    chat_model: str = "llama-3.3-70b-versatile"

    # Embeddings (local — no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dims: int = 384

    # MongoDB
    mongodb_uri: str
    mongodb_db: str = "mangrag"
    mongodb_collection: str = "documents"

    # RAG
    index_name: str = "vector_index"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    num_candidates: int = 150


settings = Settings()

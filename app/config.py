from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(..., env="AZURE_OPENAI_KEY")
    azure_openai_deployment: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_embedding_model: str = Field("text-embedding-3-large", env="AZURE_OPENAI_EMBEDDING_MODEL")

    # Azure Search
    azure_search_endpoint: str = Field(..., env="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(..., env="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field("it-kb-index", env="AZURE_SEARCH_INDEX_NAME")

    # RAG settings
    vector_top_k: int = Field(5, env="VECTOR_TOP_K")
    chunk_size: int = Field(400, env="CHUNK_SIZE")
    chunk_overlap: int = Field(80, env="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"


settings = Settings()

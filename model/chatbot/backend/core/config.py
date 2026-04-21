from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Pet Care RAG API"
    VLLM_API_URL: str = "http://localhost:8000/v1"
    CHROMA_DB_PATH: str = "../vectordb-processing/chroma_db"
    COLLECTION_NAME: str = "pet_medical_docs"
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    MAX_HISTORY_TOKENS: int = 1500
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Khởi tạo object settings để dùng chung
settings = Settings()

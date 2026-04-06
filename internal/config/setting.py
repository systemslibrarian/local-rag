from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=False)

    ollama_host: str = Field(..., validation_alias="OLLAMA_HOST")
    text_embedding_model: str = Field(..., validation_alias="TEXT_EMBEDDING_MODEL")
    llm_model: str = Field(..., validation_alias="LLM_MODEL")
    temp_folder: str = Field(..., validation_alias="TEMP_FOLDER")
    collection_name: str = Field(..., validation_alias="COLLECTION_NAME")
    pg_dsn: str = Field(..., validation_alias="PG_DSN")
    # Retrieval
    similarity_threshold: float = Field(0.30, validation_alias="SIMILARITY_THRESHOLD")
    history_window: int = Field(6, validation_alias="HISTORY_WINDOW")  # max message pairs
    # File storage (PDFs on disk instead of DB)
    file_storage_folder: str = Field("./data/files", validation_alias="FILE_STORAGE_FOLDER")
    # DB connection pool
    db_pool_size: int = Field(5, validation_alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(10, validation_alias="DB_MAX_OVERFLOW")


load_dotenv()
setting = Settings()  # type: ignore
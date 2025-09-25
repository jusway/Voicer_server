from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SQL_URL: str = "sqlite:///./app.db"
    QWEN_API_KEY: Optional[str] = None
    SILICONFLOW_KEY: Optional[str] = None

settings = Settings()


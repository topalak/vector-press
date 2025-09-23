import os
from pydantic_settings import BaseSettings
import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))

class Settings(BaseSettings):
    OLLAMA_HOST: str = ""
    GROQ_API_KEY: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    GUARDIAN_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    TAVILY_API_KEY: str = ""

    OUTPUT: str = os.path.join(ENV_FILE_DIR, 'out')
    TIME_ZONE: datetime.timezone = datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3')

    class Config:
        extra = "ignore"  # Ignore extra environment variables

        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')

settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # OpenAI API Configuration
    openai_api_key: Optional[str] = Field(
        default=None, 
        description="OpenAI API Key"
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", 
        description="OpenAI API Base URL"
    )
    
    # Model Configuration
    model: str = Field(
        default="gpt-4.1-mini", 
        description="Model name to use"
    )
    
    # VERL Configuration
    verl_api_base: str = Field(
        default="http://localhost:9999/", 
        description="VERL API Base URL"
    )
    verl_spider_data_dir: str = Field(
        default="data", 
        description="Spider data directory"
    )
    
    # WandB Configuration
    wandb_api_key: Optional[str] = Field(
        default=None, 
        description="WandB API Key (optional)"
    )


# Create a global settings instance
settings = Settings()
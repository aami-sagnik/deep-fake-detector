"""
Deep-Fake Detection System
Configuration management with environment variables and defaults.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # Model Settings
    visual_model: str = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    audio_model: str = "facebook/wav2vec2-base-960h"
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Processing Settings
    max_video_duration: int = 600  # seconds
    frame_sample_rate: int = 5  # sample every Nth frame
    audio_sample_rate: int = 16000  # Hz
    
    # Detection Thresholds
    visual_confidence_threshold: float = 0.5
    audio_confidence_threshold: float = 0.5
    biometric_confidence_threshold: float = 0.5
    metadata_confidence_threshold: float = 0.5

    # Server Settings
    gradio_server_name: str = "127.0.0.1"
    gradio_server_port: int = 7860
    gradio_share: bool = False

    # Ollama / LLM Settings
    use_ollama: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_timeout: int = 60
    
    # OpenAI Settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Paths
    cache_dir: Path = Path.home() / ".cache" / "deep-fake-detector"
    data_dir: Path = Path.home() / ".data" / "deep-fake-detector"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

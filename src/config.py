from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# dataclass = decorator (with this there is no need to add init)
@dataclass
class Config:
    # Source
    source_type: str = os.getenv("SOURCE_TYPE", "camera")
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    youtube_url: str = os.getenv("YOUTUBE_URL", "")

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Detection
    confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
    device: str = os.getenv("DETECTION_DEVICE", "cuda")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))


config = Config()
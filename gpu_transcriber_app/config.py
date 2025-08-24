from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

class FileConflictMode(Enum):
    """File conflict resolution modes."""
    ASK = "ask"
    OVERWRITE = "overwrite"
    RENAME = "rename"
    SKIP = "skip"

@dataclass
class TranscriptionConfig:
    """Configuration for transcription process."""
    model_size: str = "large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16"
    batch_size: int = 16
    beam_size: int = 5
    language: Optional[str] = None
    output_format: str = "srt"
    embed_subtitles: bool = False
    max_workers: int = 4
    disable_vad: bool = False  # Enable VAD by default for faster-whisper compatibility
    enable_diarization: bool = False
    enable_gender_classification: bool = False
    max_subtitle_duration: float = 5.0  # Maximum duration for a single subtitle segment in seconds
    max_subtitle_chars: int = 80  # Maximum characters per subtitle line
    file_conflict_mode: FileConflictMode = FileConflictMode.ASK

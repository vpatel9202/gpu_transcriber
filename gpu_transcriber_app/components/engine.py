import pathlib
import logging
from typing import List, Dict

from ..config import TranscriptionConfig
from .utils import GPUMemoryManager

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    """Handles the transcription process using optimal GPU settings."""

    def __init__(self, config: TranscriptionConfig, imports_dict=None):
        self.config = config
        self.memory_manager = GPUMemoryManager()
        self.imports = imports_dict or {}
        self.model = None
        self.batched_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transcription model based on availability and performance."""
        # Optimize batch size for RTX 4090
        optimal_batch_size = self.memory_manager.get_optimal_batch_size(self.config.model_size)
        self.config.batch_size = min(self.config.batch_size, optimal_batch_size)

        logger.info(f"Initializing model: {self.config.model_size}")
        logger.info(f"Device: {self.config.device}, Compute type: {self.config.compute_type}")
        logger.info(f"Optimal batch size: {self.config.batch_size}")

        if self.imports.get('faster_whisper') and self.config.device == "cuda":
            # Primary choice: faster-whisper for best balance
            logger.info("Using faster-whisper (CTranslate2) - optimal for RTX 4090")
            WhisperModel, BatchedInferencePipeline = self.imports['faster_whisper']
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=None,
                local_files_only=False
            )
            # Enable batched inference for maximum throughput
            self.batched_model = BatchedInferencePipeline(model=self.model)

        elif self.imports.get('openai_whisper'):
            # Fallback: OpenAI Whisper
            logger.info("Using OpenAI Whisper")
            whisper = self.imports['openai_whisper']
            self.model = whisper.load_model(self.config.model_size, device=self.config.device)

        else:
            raise RuntimeError("No transcription engine available. Install faster-whisper or openai-whisper.")

    def transcribe_audio(self, audio_path: pathlib.Path) -> List[Dict]:
        """Transcribe audio file and return segments."""
        try:
            if self.batched_model:
                # Use faster-whisper batched inference with configurable VAD
                use_vad = not getattr(self.config, 'disable_vad', True)
                logger.debug(f"Using VAD filter: {use_vad}")

                segments, info = self.batched_model.transcribe(
                    str(audio_path),
                    batch_size=self.config.batch_size,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    word_timestamps=True,
                    vad_filter=use_vad,
                    vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None
                )

                # Convert to list format
                return [
                    {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip(),
                        'words': [
                            {'start': word.start, 'end': word.end, 'word': word.word}
                            for word in (segment.words or [])
                        ] if hasattr(segment, 'words') else []
                    }
                    for segment in segments
                ]

            elif self.model is not None and hasattr(self.model, 'transcribe'):
                # OpenAI Whisper
                result = self.model.transcribe(
                    str(audio_path),
                    language=self.config.language,
                    word_timestamps=True
                )
                return result.get('segments', [])
            
            else:
                # Fallback if no transcription method is available
                logger.error("No transcription method available")
                return []

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise

        finally:
            # Clear GPU cache after each transcription
            self.memory_manager.clear_cache()

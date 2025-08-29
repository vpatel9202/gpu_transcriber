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
        
        # Validate GPU availability if requested
        if self.config.device == "cuda":
            if not self._validate_gpu():
                logger.warning("⚠️  GPU validation failed, falling back to CPU")
                self.config.device = "cpu"

        if self.imports.get('faster_whisper') and self.config.device == "cuda":
            # Check if faster-whisper has CUDA support
            has_cuda_support = self.imports.get('faster_whisper_cuda', False)
            
            if not has_cuda_support:
                logger.warning("❌ faster-whisper installed without CUDA support")
                logger.warning("💡 Install CUDA version: pip uninstall faster-whisper && pip install faster-whisper[cuda]")
                logger.info("🔄 Falling back to CPU mode")
                self.config.device = "cpu"
            else:
                # Primary choice: faster-whisper for best balance
                try:
                    logger.info("🚀 Using faster-whisper (CTranslate2) with GPU acceleration")
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
                    
                    # Verify GPU is actually being used with a test inference
                    self._verify_gpu_usage_with_test()
                    
                except Exception as e:
                    logger.error(f"❌ faster-whisper GPU initialization failed: {e}")
                    logger.info("🔄 Falling back to CPU mode")
                    self.config.device = "cpu"
                    # Retry with CPU
                    self.model = WhisperModel(
                        self.config.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=None,
                        local_files_only=False
                    )
                    self.batched_model = BatchedInferencePipeline(model=self.model)

        elif self.imports.get('faster_whisper'):
            # faster-whisper with CPU
            logger.info("🔄 Using faster-whisper with CPU")
            WhisperModel, BatchedInferencePipeline = self.imports['faster_whisper']
            self.model = WhisperModel(
                self.config.model_size,
                device="cpu",
                compute_type="int8",
                download_root=None,
                local_files_only=False
            )
            self.batched_model = BatchedInferencePipeline(model=self.model)

        elif self.imports.get('openai_whisper'):
            # Fallback: OpenAI Whisper
            try:
                if self.config.device == "cuda":
                    logger.info("🚀 Using OpenAI Whisper with GPU acceleration")
                else:
                    logger.info("🔄 Using OpenAI Whisper with CPU")
                
                whisper = self.imports['openai_whisper']
                self.model = whisper.load_model(self.config.model_size, device=self.config.device)
                
                if self.config.device == "cuda":
                    self._verify_gpu_usage()
                    
            except Exception as e:
                if self.config.device == "cuda":
                    logger.error(f"❌ OpenAI Whisper GPU initialization failed: {e}")
                    logger.info("🔄 Falling back to CPU mode")
                    whisper = self.imports['openai_whisper']
                    self.model = whisper.load_model(self.config.model_size, device="cpu")
                    self.config.device = "cpu"
                else:
                    raise e

        else:
            raise RuntimeError("No transcription engine available. Install faster-whisper or openai-whisper.")
    
    def _validate_gpu(self) -> bool:
        """Validate GPU is available and working."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("❌ CUDA is not available")
                return False
                
            if torch.cuda.device_count() == 0:
                logger.warning("❌ No CUDA devices found")
                return False
            
            # Test GPU memory allocation
            device = torch.device("cuda:0")
            test_tensor = torch.zeros(100, 100, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**2
            logger.info(f"✅ GPU validation successful: {gpu_name} ({gpu_memory}MB)")
            return True
            
        except Exception as e:
            logger.warning(f"❌ GPU validation failed: {e}")
            return False
    
    def _verify_gpu_usage_with_test(self):
        """Verify GPU usage by performing a small test inference."""
        try:
            import torch
            import numpy as np
            import tempfile
            import os
            
            if not torch.cuda.is_available():
                return
            
            # Record initial GPU memory
            initial_memory = torch.cuda.memory_allocated(0)
            
            # Create a small test audio file (1 second of silence at 16kHz)
            sample_rate = 16000
            duration = 1.0
            samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    import scipy.io.wavfile
                    scipy.io.wavfile.write(tmp_file.name, sample_rate, (samples * 32767).astype(np.int16))
                    
                    logger.info("🧪 Testing GPU usage with small inference...")
                    
                    # Perform test transcription
                    if self.batched_model:
                        # Use batched model
                        segments, info = self.batched_model.transcribe(
                            tmp_file.name,
                            batch_size=1,
                            beam_size=1,
                            word_timestamps=False
                        )
                        list(segments)  # Consume the generator
                    elif hasattr(self.model, 'transcribe'):
                        # Use regular model
                        segments, info = self.model.transcribe(
                            tmp_file.name,
                            beam_size=1,
                            word_timestamps=False
                        )
                        list(segments)  # Consume the generator
                    
                    # Check GPU memory after inference
                    post_memory = torch.cuda.memory_allocated(0)
                    memory_used = post_memory - initial_memory
                    
                    if post_memory > initial_memory:
                        logger.info(f"✅ GPU inference successful: {memory_used // 1024**2}MB allocated during test")
                    else:
                        logger.warning("⚠️  Test inference completed but no additional GPU memory allocated")
                        logger.warning("⚠️  Model may be falling back to CPU despite CUDA device setting")
                    
                except ImportError:
                    logger.warning("⚠️  scipy not available for GPU test, skipping detailed verification")
                    # Fallback to basic memory check
                    current_memory = torch.cuda.memory_allocated(0)
                    if current_memory > initial_memory:
                        logger.info(f"✅ GPU memory allocated: {current_memory // 1024**2}MB")
                    else:
                        logger.warning("⚠️  No GPU memory allocated - model may not be using GPU")
                        
                except Exception as test_error:
                    logger.warning(f"⚠️  GPU test inference failed: {test_error}")
                    logger.warning("⚠️  Model loaded but GPU usage cannot be verified")
                
                finally:
                    # Cleanup
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
            
        except Exception as e:
            logger.debug(f"Could not verify GPU usage: {e}")
    
    def _verify_gpu_usage(self):
        """Basic GPU memory check (fallback method)."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Check if there's any memory allocated on GPU
                allocated = torch.cuda.memory_allocated(0)
                if allocated > 0:
                    logger.info(f"✅ GPU memory allocated: {allocated // 1024**2}MB")
                else:
                    logger.warning("⚠️  No GPU memory allocated - model may not be using GPU")
            
        except Exception as e:
            logger.debug(f"Could not verify GPU usage: {e}")

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

import pathlib
import logging
from typing import List, Dict

import torch

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """Handles speaker diarization using pyannote.audio."""

    def __init__(self, imports_dict=None, hf_token=None):
        self.pipeline = None
        self.imports = imports_dict or {}
        self.hf_token = hf_token
        if self.imports.get('pyannote'):
            try:
                # Try different model versions
                Pipeline = self.imports['pyannote']
                models_to_try = [
                    "pyannote/speaker-diarization-3.1",
                    "pyannote/speaker-diarization",
                    "pyannote/speaker-diarization@2022.07"
                ]

                for model_name in models_to_try:
                    try:
                        logger.info(f"Attempting to load {model_name}")
                        
                        # Determine device for diarization
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda":
                            try:
                                # Test GPU allocation
                                test_tensor = torch.zeros(10, 10, device="cuda")
                                del test_tensor
                                torch.cuda.empty_cache()
                                logger.info(f"🚀 Loading diarization model on GPU")
                            except Exception as gpu_error:
                                logger.warning(f"⚠️  GPU test failed for diarization: {gpu_error}")
                                device = "cpu"
                                logger.info(f"🔄 Falling back to CPU for diarization")
                        else:
                            logger.info(f"🔄 Loading diarization model on CPU")
                        
                        # Load with device specification
                        self.pipeline = Pipeline.from_pretrained(
                            model_name, 
                            use_auth_token=self.hf_token
                        )
                        
                        # Move to device if CUDA is available
                        if hasattr(self.pipeline, 'to') and device == "cuda":
                            try:
                                self.pipeline.to(torch.device("cuda"))
                                logger.info(f"✅ Diarization model moved to GPU")
                            except Exception as e:
                                logger.warning(f"⚠️  Could not move diarization model to GPU: {e}")
                        
                        logger.info(f"✅ Successfully loaded {model_name} on {device.upper()}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load {model_name}: {e}")
                        continue

                if not self.pipeline:
                    logger.warning("Could not load any speaker diarization model")
                    logger.info("Speaker diarization will be disabled. You may need a HuggingFace token for some models.")

            except Exception as e:
                logger.warning(f"Speaker diarization initialization failed: {e}")
                logger.info("Speaker diarization will be disabled.")

    def diarize(self, audio_path: pathlib.Path) -> List[Dict]:
        """Perform speaker diarization on audio file."""
        if not self.pipeline:
            logger.debug("No diarization pipeline available")
            return []

        try:
            logger.debug(f"Running diarization on {audio_path}")

            # Check if audio file exists and is readable
            if not audio_path.exists():
                logger.error(f"Audio file does not exist: {audio_path}")
                return []

            # Check file size
            file_size = audio_path.stat().st_size
            if file_size == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return []

            logger.debug(f"Audio file size: {file_size / 1024 / 1024:.2f} MB")

            # Run diarization with more detailed error handling
            diarization = self.pipeline(str(audio_path))

            segments = []
            segment_count = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
                segment_count += 1

            logger.debug(f"Diarization found {segment_count} segments")
            return segments

        except Exception as e:
            logger.error(f"Speaker diarization failed with error: {e}")
            logger.debug(f"Diarization error details: {type(e).__name__}: {str(e)}")
            return []

class GenderClassifier:
    """Classifies speaker gender from audio segments."""

    def __init__(self, imports_dict=None):
        self.classifier = None
        self.use_librosa = False
        self.model_type = None
        self.device = None
        self.imports = imports_dict or {}
        self.logger = logging.getLogger(__name__)

        # Try to load SpeechBrain gender classifier first
        if self.imports.get('speechbrain'):
            try:
                # Use SpeechBrain's pre-trained gender classification model
                EncoderClassifier = self.imports['speechbrain']
                
                # Try different gender classification models in order of preference
                models_to_try = [
                    ("speechbrain/lang-id-commonlanguage_ecapa", "pretrained_models/lang-id-commonlanguage_ecapa"),
                    ("speechbrain/spkrec-ecapa-voxceleb", "pretrained_models/spkrec-ecapa-voxceleb")
                ]
                
                for model_source, save_dir in models_to_try:
                    try:
                        # Determine device and validate GPU if needed
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if device == "cuda":
                            try:
                                # Test GPU allocation
                                test_tensor = torch.zeros(10, 10, device="cuda")
                                del test_tensor
                                torch.cuda.empty_cache()
                                self.logger.info(f"🚀 Loading SpeechBrain model on GPU: {model_source}")
                            except Exception as gpu_error:
                                self.logger.warning(f"⚠️  GPU test failed for SpeechBrain: {gpu_error}")
                                device = "cpu"
                                self.logger.info(f"🔄 Falling back to CPU for SpeechBrain: {model_source}")
                        else:
                            self.logger.info(f"🔄 Loading SpeechBrain model on CPU: {model_source}")
                        
                        self.classifier = EncoderClassifier.from_hparams(
                            source=model_source,
                            savedir=save_dir,
                            run_opts={"device": device}
                        )
                        self.model_type = "speechbrain_ecapa"
                        self.device = device
                        self.logger.info(f"✅ Loaded SpeechBrain gender classifier on {device.upper()}: {model_source}")
                        break
                    except Exception as model_error:
                        self.logger.debug(f"Failed to load {model_source}: {model_error}")
                        continue
                
                if not hasattr(self, 'classifier') or self.classifier is None:
                    raise Exception("No SpeechBrain model could be loaded")
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Failed to load SpeechBrain classifier: {e}")
                
                # Check if this is a Windows symlink permission issue
                if "WinError 1314" in error_msg or "required privilege is not held" in error_msg.lower():
                    import platform
                    if platform.system() == "Windows":
                        self.logger.warning("⚠️  SpeechBrain requires elevated privileges on Windows for symlink creation.")
                        self.logger.warning("💡 Solutions:")
                        self.logger.warning("   1. Run this terminal as Administrator, or")
                        self.logger.warning("   2. Enable Windows Developer Mode (Settings > Update & Security > For developers)")
                        self.logger.warning("   3. Gender classification will fall back to librosa-only mode (less accurate)")
                
                self.classifier = None

        # Fallback to librosa-based approach
        if not self.classifier and self.imports.get('librosa'):
            self.use_librosa = True
            self.logger.info("Using librosa-based gender classification")

        if not self.classifier and not self.use_librosa:
            self.logger.warning("No gender classification method available")

    def classify_gender(self, audio_path: pathlib.Path, start_time: float, end_time: float) -> str:
        """Classify gender for a specific audio segment."""
        try:
            if self.use_librosa:
                return self._classify_with_librosa(audio_path, start_time, end_time)
            elif self.classifier:
                return self._classify_with_speechbrain(audio_path, start_time, end_time)
            else:
                return "UNKNOWN"
        except Exception as e:
            self.logger.warning(f"Gender classification failed: {e}")
            return "UNKNOWN"

    def _classify_with_librosa(self, audio_path: pathlib.Path, start_time: float, end_time: float) -> str:
        """Classify gender using librosa features (pitch, spectral features)."""
        try:
            # Load audio segment
            librosa = self.imports.get('librosa')
            np = self.imports.get('numpy')
            if not librosa or not np:
                return "UNKNOWN"

            y, sr = librosa.load(str(audio_path), offset=start_time, duration=end_time-start_time)

            if len(y) < sr * 0.5:  # Need at least 0.5 seconds
                return "UNKNOWN"

            # Extract features
            # 1. Fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)

            # Get average pitch
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if not pitch_values:
                return "UNKNOWN"

            avg_pitch = np.mean(pitch_values)

            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)

            # 3. MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)

            # Simple classification based on voice characteristics
            # These thresholds are approximate and may need tuning
            confidence = 0.0

            # Pitch-based classification (Hz)
            if avg_pitch < 165:  # Typically male range
                confidence += 0.4
                gender_vote = "MALE"
            elif avg_pitch > 200:  # Typically female range
                confidence += 0.4
                gender_vote = "FEMALE"
            else:  # Ambiguous range
                confidence += 0.1
                gender_vote = "UNKNOWN"

            # Spectral centroid (Hz) - females typically have higher values
            if avg_spectral_centroid > 3000:
                if gender_vote == "FEMALE":
                    confidence += 0.3
                elif gender_vote == "MALE":
                    confidence -= 0.2
                else:
                    gender_vote = "FEMALE"
                    confidence += 0.2
            elif avg_spectral_centroid < 2000:
                if gender_vote == "MALE":
                    confidence += 0.3
                elif gender_vote == "FEMALE":
                    confidence -= 0.2
                else:
                    gender_vote = "MALE"
                    confidence += 0.2

            # First MFCC coefficient (related to overall spectral shape)
            if mfcc_means[0] > -20:  # Higher values often indicate female voices
                if gender_vote == "FEMALE":
                    confidence += 0.2
                elif gender_vote == "MALE":
                    confidence -= 0.1

            # Return result based on confidence
            if confidence >= 0.5:
                return gender_vote
            else:
                return "UNKNOWN"

        except Exception as e:
            self.logger.warning(f"Librosa gender classification failed: {e}")
            return "UNKNOWN"

    def _classify_with_speechbrain(self, audio_path: pathlib.Path, start_time: float, end_time: float) -> str:
        """Classify gender using SpeechBrain model."""
        try:
            if not self.classifier:
                return "UNKNOWN"
            
            # Load and process audio segment
            import torchaudio
            
            # Load the specific audio segment
            waveform, sample_rate = torchaudio.load(
                str(audio_path), 
                frame_offset=int(start_time * 16000), 
                num_frames=int((end_time - start_time) * 16000)
            )
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure minimum length (at least 1 second)
            min_length = 16000  # 1 second at 16kHz
            if waveform.shape[1] < min_length:
                # Pad with zeros if too short
                pad_length = min_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # Extract embeddings using the ECAPA model
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(waveform)
                
            # Use a simple heuristic based on embedding characteristics for gender classification
            # This is a simplified approach - in practice, you'd train a classifier on top of embeddings
            embedding_vector = embeddings.squeeze().cpu().numpy()
            
            # Simple gender classification based on embedding statistics
            # These thresholds are heuristic and may need tuning with real data
            mean_embedding = float(embedding_vector.mean())
            std_embedding = float(embedding_vector.std())
            
            # Heuristic classification based on embedding characteristics
            # Female voices tend to have different embedding patterns
            if mean_embedding > 0.1 and std_embedding > 0.15:
                gender = "FEMALE"
                confidence = min(0.8, abs(mean_embedding) + std_embedding)
            elif mean_embedding < -0.1 and std_embedding < 0.12:
                gender = "MALE"
                confidence = min(0.8, abs(mean_embedding) + (0.15 - std_embedding))
            else:
                # Fall back to librosa-based classification for ambiguous cases
                self.logger.debug("SpeechBrain classification ambiguous, falling back to librosa")
                return self._classify_with_librosa(audio_path, start_time, end_time)
            
            self.logger.debug(f"SpeechBrain gender classification: {gender} (confidence: {confidence:.2f})")
            return gender if confidence > 0.4 else "UNKNOWN"
            
        except ImportError:
            self.logger.warning("torchaudio not available, falling back to librosa")
            return self._classify_with_librosa(audio_path, start_time, end_time)
        except Exception as e:
            self.logger.warning(f"SpeechBrain gender classification failed: {e}")
            # Fall back to librosa method
            return self._classify_with_librosa(audio_path, start_time, end_time)

    def classify_speakers_gender(self, audio_path: pathlib.Path,
                               speaker_segments: List[Dict]) -> Dict[str, str]:
        """Classify gender for each unique speaker."""
        speaker_genders = {}

        # Group segments by speaker
        speaker_times = {}
        for segment in speaker_segments:
            speaker = segment.get('speaker')
            if speaker:
                if speaker not in speaker_times:
                    speaker_times[speaker] = []
                speaker_times[speaker].append((segment['start'], segment['end']))

        # Classify each speaker using their longest segment
        for speaker, times in speaker_times.items():
            # Find the longest segment for this speaker
            longest_segment = max(times, key=lambda x: x[1] - x[0])
            start_time, end_time = longest_segment

            # Only classify if segment is long enough (at least 1 second)
            if end_time - start_time >= 1.0:
                gender = self.classify_gender(audio_path, start_time, end_time)
                speaker_genders[speaker] = gender
                self.logger.info(f"Classified {speaker} as {gender}")
            else:
                speaker_genders[speaker] = "UNKNOWN"
                self.logger.info(f"Segment too short for {speaker}, marking as UNKNOWN")

        return speaker_genders

    def classify_entire_audio(self, audio_path: pathlib.Path, duration: float | None = None) -> str:
        """Classify gender for entire audio file (when no diarization available)."""
        try:
            # Use the first 30 seconds of audio for classification (or entire file if shorter)
            max_duration = min(30.0, duration) if duration else 30.0
            return self.classify_gender(audio_path, 0.0, max_duration)
        except Exception as e:
            self.logger.warning(f"Entire audio gender classification failed: {e}")
            return "UNKNOWN"

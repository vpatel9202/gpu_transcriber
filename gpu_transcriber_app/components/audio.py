import pathlib
import logging
from typing import List, Dict

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
                        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=self.hf_token)
                        logger.info(f"Successfully loaded {model_name}")
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
        self.imports = imports_dict or {}
        self.logger = logging.getLogger(__name__)

        # Try to load SpeechBrain gender classifier first
        if self.imports.get('speechbrain'):
            try:
                # Use SpeechBrain's pre-trained gender classification model
                EncoderClassifier = self.imports['speechbrain']
                
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"collect_in": None}
                )
                self.logger.info("Loaded SpeechBrain gender classifier")
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
            # This would need a specific gender classification model
            # SpeechBrain's spkrec models are for speaker recognition, not gender
            # For now, return UNKNOWN as we'd need a specific gender model
            self.logger.debug("SpeechBrain gender classification not implemented yet")
            return "UNKNOWN"
        except Exception as e:
            self.logger.warning(f"SpeechBrain gender classification failed: {e}")
            return "UNKNOWN"

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

import time
import pathlib
import logging
from typing import List, Dict, Tuple

from .config import TranscriptionConfig
from .components.video import VideoProcessor
from .components.audio import SpeakerDiarizer, GenderClassifier
from .components.engine import TranscriptionEngine
from .components.subtitles import SubtitleGenerator, SubtitleEmbedder
from .components.utils import ProgressTracker, FileHandler

logger = logging.getLogger(__name__)

class BulkTranscriber:
    """Main class for bulk video transcription."""

    def __init__(self, config: TranscriptionConfig, imports_dict=None, hf_token=None):
        self.config = config
        self.imports = imports_dict or {}
        self.hf_token = hf_token
        self.processor = VideoProcessor()
        self.engine = TranscriptionEngine(config, imports_dict)
        self.subtitle_generator = SubtitleGenerator()
        self.embedder = SubtitleEmbedder()
        self.diarizer = SpeakerDiarizer(imports_dict, hf_token) if config.enable_diarization else None
        self.gender_classifier = GenderClassifier(imports_dict) if config.enable_gender_classification else None
        self.progress_tracker = ProgressTracker(logger)
        self.file_handler = FileHandler(config)
        self.temp_dir = pathlib.Path("temp_transcription")
        self.temp_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.start_time = time.time()
        self.processed_count = 0
        self.failed_count = 0

    def get_video_files(self, input_path: pathlib.Path) -> List[pathlib.Path]:
        """Get list of video files to process."""
        if input_path.is_file():
            if self.processor.is_video_file(input_path):
                return [input_path]
            else:
                raise ValueError(f"File {input_path} is not a supported video format")

        elif input_path.is_dir():
            video_files = []
            for ext in self.processor.SUPPORTED_FORMATS:
                video_files.extend(input_path.glob(f"**/*{ext}"))
                video_files.extend(input_path.glob(f"**/*{ext.upper()}"))

            if not video_files:
                raise ValueError(f"No video files found in directory {input_path}")

            return sorted(video_files)

        else:
            raise ValueError(f"Input path {input_path} does not exist")

    def transcribe_single_video(self, video_path: pathlib.Path) -> Tuple[bool, str, Dict]:
        """Transcribe a single video file with detailed error tracking."""
        result_info = {
            'video_path': str(video_path),
            'output_files': [],
            'errors': [],
            'processing_time': 0,
            'audio_duration': 0,
            'segments_count': 0
        }

        start_time = time.time()

        try:
            logger.info(f"Starting transcription: {video_path}")
            logger.debug(f"Video file size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")

            # Check if output already exists and handle file conflicts
            output_paths = self._get_output_paths(video_path)
            logger.debug(f"Expected output paths: {list(output_paths.values())}")

            # Handle file conflicts using the file handler
            final_output_paths = {}
            for format_name, path in output_paths.items():
                should_proceed, final_path = self.file_handler.handle_existing_file(path)
                if not should_proceed:
                    logger.info(f"Skipping {format_name} file due to user choice")
                    continue
                final_output_paths[format_name] = final_path

            if not final_output_paths:
                logger.info("All outputs skipped by user choice")
                return True, "Skipped (user choice)", result_info

            # Get video information
            try:
                video_info = self.processor.get_video_info(video_path)
                if video_info:
                    logger.debug(f"Video info: {video_info}")
                    result_info['audio_duration'] = video_info.get('duration', 0)
            except Exception as e:
                logger.warning(f"Could not get video info: {e}")
                result_info['errors'].append(f"Video info error: {e}")

            # Extract audio
            self.progress_tracker.start_operation("Extracting audio from video")
            try:
                audio_path = self.processor.extract_audio(video_path, self.temp_dir)
                self.progress_tracker.update_progress(f"Extracted to {audio_path.name}")
                self.progress_tracker.finish_operation(True)
                logger.info(f"Audio extracted to: {audio_path}")
                logger.debug(f"Audio file size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")
            except Exception as e:
                self.progress_tracker.finish_operation(False)
                error_msg = f"Audio extraction failed: {e}"
                logger.error(error_msg)
                result_info['errors'].append(error_msg)
                return False, error_msg, result_info

            # Transcribe
            self.progress_tracker.start_operation("Transcribing audio with Whisper")
            try:
                segments = self.engine.transcribe_audio(audio_path)
                result_info['segments_count'] = len(segments) if segments else 0
                self.progress_tracker.update_progress(f"Found {result_info['segments_count']} segments")
                self.progress_tracker.finish_operation(True)
                logger.info(f"Transcription complete. Found {result_info['segments_count']} segments")

                if segments:
                    # Log first few segments for verification
                    for i, segment in enumerate(segments[:3]):
                        logger.debug(f"Segment {i+1}: {segment['start']:.2f}-{segment['end']:.2f}s: {segment['text'][:100]}...")

                    # Perform speaker diarization if enabled
                    diarization_segments = []
                    if self.config.enable_diarization and self.diarizer:
                        self.progress_tracker.start_operation("Analyzing speakers (diarization)")
                        try:
                            diarization_segments = self.diarizer.diarize(audio_path)
                            if diarization_segments:
                                self.progress_tracker.update_progress(f"Found {len(diarization_segments)} speaker segments")
                                segments = self.subtitle_generator.assign_speakers(segments, diarization_segments)
                                self.progress_tracker.finish_operation(True)
                                logger.info(f"Found {len(diarization_segments)} speaker segments")
                            else:
                                self.progress_tracker.finish_operation(False)
                                logger.warning("Speaker diarization found no segments")
                        except Exception as e:
                            self.progress_tracker.finish_operation(False)
                            logger.warning(f"Speaker diarization failed: {e}")

                    # Perform gender classification if enabled
                    if self.config.enable_gender_classification and self.gender_classifier:
                        self.progress_tracker.start_operation("Classifying speaker genders")
                        try:
                            # Try speaker-based classification first if diarization worked
                            if diarization_segments:
                                speaker_genders = self.gender_classifier.classify_speakers_gender(audio_path, diarization_segments)
                                if speaker_genders:
                                    self.progress_tracker.update_progress(f"Classified {len(speaker_genders)} speakers")
                                    segments = self.subtitle_generator.apply_gender_labels(segments, speaker_genders)
                                    self.progress_tracker.finish_operation(True)
                                    logger.info(f"Gender classification results: {speaker_genders}")
                                else:
                                    self.progress_tracker.finish_operation(False)
                                    logger.warning("Speaker-based gender classification found no results")
                            else:
                                # Fallback: classify entire audio as single speaker
                                logger.info("No speaker diarization available, classifying entire audio")
                                audio_duration = result_info.get('audio_duration', None)
                                overall_gender = self.gender_classifier.classify_entire_audio(audio_path, audio_duration)

                                if overall_gender != "UNKNOWN":
                                    self.progress_tracker.update_progress(f"Classified overall gender as {overall_gender}")
                                    # Apply gender to all segments
                                    for segment in segments:
                                        segment['gender'] = overall_gender
                                    self.progress_tracker.finish_operation(True)
                                    logger.info(f"Overall gender classification: {overall_gender}")
                                else:
                                    self.progress_tracker.finish_operation(False)
                                    logger.warning("Could not determine gender from audio")
                        except Exception as e:
                            self.progress_tracker.finish_operation(False)
                            logger.warning(f"Gender classification failed: {e}")

                    # Break long segments into shorter, more readable ones
                    logger.info("Breaking long segments for better readability...")
                    original_count = len(segments)
                    segments = self.subtitle_generator.break_long_segments(
                        segments,
                        max_duration=self.config.max_subtitle_duration,
                        max_chars=self.config.max_subtitle_chars
                    )
                    logger.info(f"Segment processing complete: {original_count} → {len(segments)} segments")

            except Exception as e:
                error_msg = f"Transcription failed: {e}"
                logger.error(error_msg, exc_info=True)
                result_info['errors'].append(error_msg)
                return False, error_msg, result_info

            if not segments:
                error_msg = "No speech detected in audio"
                logger.warning(error_msg)
                result_info['errors'].append(error_msg)
                return False, error_msg, result_info

            # Generate subtitle files
            logger.info("Generating output files...")
            try:
                success, output_files = self._generate_outputs(video_path, segments, final_output_paths)
                result_info['output_files'] = output_files

                if success:
                    logger.info("Output generation successful")
                    for file_path in output_files:
                        if pathlib.Path(file_path).exists():
                            file_size = pathlib.Path(file_path).stat().st_size
                            logger.info(f"Created: {file_path} ({file_size} bytes)")
                        else:
                            logger.warning(f"Expected output file not found: {file_path}")
                            result_info['errors'].append(f"Output file not created: {file_path}")
                else:
                    error_msg = "Output generation failed"
                    logger.error(error_msg)
                    result_info['errors'].append(error_msg)
                    return False, error_msg, result_info

            except Exception as e:
                error_msg = f"Output generation error: {e}"
                logger.error(error_msg, exc_info=True)
                result_info['errors'].append(error_msg)
                return False, error_msg, result_info

            # Cleanup temporary files
            try:
                if audio_path.exists():
                    audio_path.unlink()
                    logger.debug(f"Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temp file {audio_path}: {e}")

            self.processed_count += 1
            result_info['processing_time'] = time.time() - start_time

            logger.info(f"Successfully processed {video_path.name} in {result_info['processing_time']:.2f}s")
            return True, "Success", result_info

        except Exception as e:
            error_msg = f"Unexpected error processing {video_path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            result_info['errors'].append(error_msg)
            result_info['processing_time'] = time.time() - start_time
            self.failed_count += 1
            return False, error_msg, result_info

    def _get_output_paths(self, video_path: pathlib.Path) -> Dict[str, pathlib.Path]:
        """Get output file paths for different formats."""
        base_path = video_path.parent / video_path.stem

        paths = {
            'srt': base_path.with_suffix('.srt'),
            'vtt': base_path.with_suffix('.vtt'),
            'txt': base_path.with_suffix('.txt'),
            'json': base_path.with_suffix('.json')
        }

        if self.config.embed_subtitles:
            paths['embedded'] = base_path.with_suffix(f'_subtitled{video_path.suffix}')

        return paths

    def _generate_outputs(self, video_path: pathlib.Path, segments: List[Dict],
                         output_paths: Dict[str, pathlib.Path]) -> Tuple[bool, List[str]]:
        """Generate all requested output formats with detailed tracking."""
        created_files = []

        try:
            logger.debug(f"Generating {self.config.output_format} output for {video_path.name}")

            # Generate subtitle file in requested format
            content = ""
            subtitle_file = None

            if self.config.output_format == "srt":
                content = self.subtitle_generator.generate_srt(segments)
                subtitle_file = output_paths['srt']

            elif self.config.output_format == "vtt":
                content = self.subtitle_generator.generate_vtt(segments)
                subtitle_file = output_paths['vtt']

            elif self.config.output_format == "txt":
                content = self.subtitle_generator.generate_text(segments)
                subtitle_file = output_paths['txt']

            elif self.config.output_format == "json":
                content = self.subtitle_generator.generate_json(segments)
                subtitle_file = output_paths['json']

            else:
                # Default to SRT
                logger.warning(f"Unknown format {self.config.output_format}, defaulting to SRT")
                content = self.subtitle_generator.generate_srt(segments)
                subtitle_file = output_paths['srt']

            # Write the subtitle file
            if subtitle_file and content:
                try:
                    logger.debug(f"Writing subtitle file: {subtitle_file}")
                    logger.debug(f"Content length: {len(content)} characters")

                    # Ensure parent directory exists
                    subtitle_file.parent.mkdir(parents=True, exist_ok=True)

                    # Write with explicit encoding
                    subtitle_file.write_text(content, encoding='utf-8')

                    # Verify file was created and has content
                    if subtitle_file.exists():
                        file_size = subtitle_file.stat().st_size
                        logger.info(f"✅ Created subtitle file: {subtitle_file} ({file_size} bytes)")
                        created_files.append(str(subtitle_file))

                        # Log first few lines for verification
                        if file_size > 0:
                            with open(subtitle_file, 'r', encoding='utf-8') as f:
                                first_lines = [f.readline().strip() for _ in range(3)]
                            logger.debug(f"First lines of {subtitle_file.name}: {first_lines}")
                        else:
                            logger.warning(f"Subtitle file {subtitle_file} is empty!")
                    else:
                        logger.error(f"❌ Subtitle file was not created: {subtitle_file}")
                        return False, created_files

                except Exception as e:
                    logger.error(f"Failed to write subtitle file {subtitle_file}: {e}", exc_info=True)
                    return False, created_files

            # Embed subtitles if requested
            if self.config.embed_subtitles and subtitle_file and subtitle_file.exists():
                if self.config.output_format in ['srt', 'vtt']:  # Only embed formats that support it
                    logger.info("Embedding subtitles into video...")
                    try:
                        embed_success = self.embedder.embed_soft_subtitles(
                            video_path, subtitle_file, output_paths['embedded'],
                            self.config.language or 'en'
                        )
                        if embed_success and output_paths['embedded'].exists():
                            logger.info(f"✅ Created video with subtitles: {output_paths['embedded']}")
                            created_files.append(str(output_paths['embedded']))
                        else:
                            logger.warning(f"❌ Failed to embed subtitles for {video_path.name}")
                    except Exception as e:
                        logger.error(f"Subtitle embedding error: {e}", exc_info=True)
                else:
                    logger.info(f"Skipping subtitle embedding for format: {self.config.output_format}")

            return True, created_files

        except Exception as e:
            logger.error(f"Failed to generate outputs for {video_path.name}: {e}", exc_info=True)
            return False, created_files

    def process_videos(self, input_path: pathlib.Path) -> Dict:
        """Process all videos in the input path with detailed reporting."""
        video_files = self.get_video_files(input_path)

        logger.info("="*60)
        logger.info("🚀 STARTING BULK TRANSCRIPTION")
        logger.info("="*60)
        logger.info(f"📂 Input path: {input_path}")
        logger.info(f"🎬 Found {len(video_files)} video files to process")
        logger.info(f"🤖 Using model: {self.config.model_size}")
        logger.info(f"📝 Output format: {self.config.output_format}")
        logger.info(f"💾 GPU Memory: {self.engine.memory_manager.total_memory}MB")
        logger.info(f"🔧 Embed subtitles: {self.config.embed_subtitles}")
        logger.info(f"📄 Log file: transcription_detailed.log")
        logger.info("-"*60)

        # Detailed processing results
        processing_results = []
        all_created_files = []

        # Process videos with progress bar
        from tqdm import tqdm
        with tqdm(total=len(video_files), desc="Transcribing videos", ncols=100) as pbar:
            for video_file in video_files:
                success, message, result_info = self.transcribe_single_video(video_file)

                # Store detailed results
                processing_results.append({
                    'video': str(video_file),
                    'success': success,
                    'message': message,
                    'result_info': result_info
                })

                # Collect all created files
                all_created_files.extend(result_info.get('output_files', []))

                # Update progress bar
                status = "✓" if success else "✗"
                pbar.set_postfix(
                    current=video_file.name[:25],
                    status=f"{status} {message[:15]}",
                    success=self.processed_count,
                    failed=self.failed_count
                )
                pbar.update(1)

                # Memory management
                if self.processed_count % 10 == 0:
                    self.engine.memory_manager.clear_cache()

        # Comprehensive cleanup of temp files and symlinks
        self._cleanup_all_temp_files()

        # Generate detailed summary

        # Generate detailed summary
        total_time = time.time() - self.start_time

        # Log all created files
        if all_created_files:
            logger.info("\n" + "="*60)
            logger.info("📁 CREATED OUTPUT FILES:")
            logger.info("="*60)
            for file_path in sorted(set(all_created_files)):  # Remove duplicates and sort
                if pathlib.Path(file_path).exists():
                    file_size = pathlib.Path(file_path).stat().st_size
                    logger.info(f"✅ {file_path} ({file_size:,} bytes)")
                else:
                    logger.warning(f"❌ {file_path} (FILE NOT FOUND)")

        # Log errors if any
        all_errors = []
        for result in processing_results:
            if result['result_info'].get('errors'):
                all_errors.extend(result['result_info']['errors'])

        if all_errors:
            logger.info("\n" + "="*60)
            logger.info("⚠️  ERRORS ENCOUNTERED:")
            logger.info("="*60)
            for error in all_errors:
                logger.warning(f"❌ {error}")

        # Return comprehensive summary
        return {
            'processed': self.processed_count,
            'failed': self.failed_count,
            'total_time': total_time,
            'avg_time_per_video': total_time / len(video_files) if video_files else 0,
            'processing_results': processing_results,
            'created_files': sorted(set(all_created_files)),
            'all_errors': all_errors
        }

    def _cleanup_all_temp_files(self):
        """Comprehensive cleanup of temporary files and symlinks."""
        import shutil
        import platform
        
        cleanup_paths = []
        
        # 1. Clean up temp transcription directory
        if self.temp_dir.exists():
            cleanup_paths.append(("Temp audio files", self.temp_dir))
        
        # 2. Clean up any broken symlinks in pretrained_models
        pretrained_dir = pathlib.Path("pretrained_models")
        if pretrained_dir.exists():
            cleanup_paths.append(("Pretrained models cache", pretrained_dir))
        
        # 3. On Windows, also check for orphaned symlinks
        if platform.system() == "Windows":
            # Look for common cache directories that might have broken symlinks
            possible_cache_dirs = [
                pathlib.Path.home() / ".cache" / "huggingface",
                pathlib.Path.home() / ".cache" / "speechbrain_models",
            ]
            for cache_dir in possible_cache_dirs:
                if cache_dir.exists():
                    cleanup_paths.append(("HF/SpeechBrain cache", cache_dir))
        
        for description, path in cleanup_paths:
            try:
                if path == self.temp_dir:
                    # Special handling for temp directory - remove all contents
                    if path.exists():
                        logger.debug(f"Cleaning up {description}: {path}")
                        shutil.rmtree(path, ignore_errors=True)
                        logger.info(f"✅ Cleaned up {description}")
                else:
                    # For other directories, just clean broken symlinks and temp files
                    self._clean_broken_symlinks(path, description)
                        
            except Exception as e:
                logger.warning(f"Could not clean up {description} at {path}: {e}")
    
    def _clean_broken_symlinks(self, directory: pathlib.Path, description: str):
        """Remove broken symlinks from a directory."""
        if not directory.exists():
            return
            
        broken_links = []
        temp_files = []
        
        try:
            for item in directory.rglob("*"):
                if item.is_symlink() and not item.exists():
                    # Broken symlink
                    broken_links.append(item)
                elif item.name.startswith(('.tmp', 'tmp_', 'temp_')):
                    # Temp files
                    temp_files.append(item)
            
            # Remove broken symlinks
            for link in broken_links:
                try:
                    link.unlink()
                    logger.debug(f"Removed broken symlink: {link}")
                except Exception as e:
                    logger.debug(f"Could not remove broken symlink {link}: {e}")
            
            # Remove temp files
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file, ignore_errors=True)
                    logger.debug(f"Removed temp file: {temp_file}")
                except Exception as e:
                    logger.debug(f"Could not remove temp file {temp_file}: {e}")
            
            if broken_links or temp_files:
                logger.info(f"✅ Cleaned up {len(broken_links)} broken symlinks and {len(temp_files)} temp files from {description}")
                
        except Exception as e:
            logger.debug(f"Error during cleanup of {description}: {e}")

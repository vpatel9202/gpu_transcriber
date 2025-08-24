import pathlib
import ffmpeg
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video file processing and audio extraction."""

    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}

    @staticmethod
    def is_video_file(file_path: pathlib.Path) -> bool:
        """Check if file is a supported video format."""
        return file_path.suffix.lower() in VideoProcessor.SUPPORTED_FORMATS

    @staticmethod
    def extract_audio(video_path: pathlib.Path, temp_dir: pathlib.Path) -> pathlib.Path:
        """Extract audio from video file."""
        audio_path = temp_dir / f"{video_path.stem}_audio.wav"

        try:
            # Convert paths to strings to handle network drives properly
            video_str = str(video_path)
            audio_str = str(audio_path)

            (
                ffmpeg
                .input(video_str)
                .output(audio_str,
                       acodec='pcm_s16le',
                       ac=1,  # mono
                       ar='16000')  # 16kHz sample rate
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"Audio extraction failed for {video_path}: {e.stderr.decode()}")
            raise

    @staticmethod
    def get_video_info(video_path: pathlib.Path) -> Dict:
        """Get video metadata."""
        try:
            # Convert to string to handle network paths
            video_str = str(video_path)
            probe = ffmpeg.probe(video_str)
            video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            # Safely parse frame rate (e.g., "30/1" or "30000/1001")
            fps_str = video_info['r_frame_rate']
            if '/' in fps_str:
                numerator, denominator = fps_str.split('/')
                fps = float(numerator) / float(denominator)
            else:
                fps = float(fps_str)

            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': fps,
                'codec': video_info['codec_name']
            }
        except Exception as e:
            logger.warning(f"Could not get video info for {video_path}: {e}")
            return {}

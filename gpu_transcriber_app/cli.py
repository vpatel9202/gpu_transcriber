import sys
import click
import torch
import psutil
import pathlib
import logging

from .config import TranscriptionConfig, FileConflictMode
from .transcriber import BulkTranscriber
from .components.utils import setup_logging


def check_dependencies():
    """Check and report missing dependencies with clear installation instructions."""
    missing_deps = []

    # Check for transcription engines
    transcription_available = False
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        transcription_available = True
    except ImportError:
        try:
            import whisper
            transcription_available = True
        except ImportError:
            pass

    if not transcription_available:
        missing_deps.append("Transcription engine: pip install faster-whisper OR pip install openai-whisper")

    return missing_deps

def import_optional_dependencies(enable_diarization: bool = False, enable_gender: bool = False, hf_token: str | None = None):
    """Import optional dependencies only when needed."""
    imports = {}

    # Transcription engines
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        imports['faster_whisper'] = (WhisperModel, BatchedInferencePipeline)
    except ImportError:
        imports['faster_whisper'] = None

    try:
        import whisper
        imports['openai_whisper'] = whisper
    except ImportError:
        imports['openai_whisper'] = None

    # Speaker diarization
    if enable_diarization:
        try:
            from pyannote.audio import Pipeline
            imports['pyannote'] = Pipeline
        except ImportError:
            click.echo("❌ ERROR: Speaker diarization requested but pyannote.audio not installed")
            click.echo("Install with: pip install pyannote.audio")
            sys.exit(1)

    # Gender classification
    if enable_gender:
        try:
            import librosa
            import numpy as np
            imports['librosa'] = librosa
            imports['numpy'] = np
        except ImportError:
            click.echo("❌ ERROR: Gender classification requested but required libraries not installed")
            click.echo("Install with: pip install librosa numpy")
            sys.exit(1)

        # SpeechBrain is optional for gender classification
        try:
            # Use the new import path
            from speechbrain.inference.classifiers import EncoderClassifier
            imports['speechbrain'] = EncoderClassifier
        except ImportError:
            try:
                # Fallback to old import for backwards compatibility
                from speechbrain.pretrained import EncoderClassifier  # type: ignore[import-untyped]
                imports['speechbrain'] = EncoderClassifier
            except ImportError:
                imports['speechbrain'] = None
                click.echo("ℹ️  Note: SpeechBrain not available, using librosa-only gender classification")

    return imports

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='large-v3',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'turbo']),
              help='Whisper model to use (larger = more accurate but slower)')
@click.option('--language', '-l', default=None, help='Force language (e.g., "en", "es", "fr")')
@click.option('--output-format', '-f', default='srt',
              type=click.Choice(['srt', 'vtt', 'txt', 'json']),
              help='Output format for transcriptions')
@click.option('--embed/--no-embed', default=False,
              help='Embed subtitles into video files as soft subtitle streams')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cuda', 'cpu']),
              help='Processing device')
@click.option('--batch-size', '-b', default=16,
              help='Batch size for GPU processing (auto-optimized for RTX 4090)')
@click.option('--compute-type', default='float16',
              type=click.Choice(['float16', 'float32', 'int8']),
              help='Compute precision (float16 recommended for RTX 4090)')
@click.option('--disable-vad/--enable-vad', default=False,
              help='Disable Voice Activity Detection (useful for music/mixed content)')
@click.option('--enable-diarization', is_flag=True, default=False,
              help='Enable speaker diarization to identify different speakers')
@click.option('--enable-gender-classification', is_flag=True, default=False,
              help='Enable gender classification (works with or without diarization)')
@click.option('--max-subtitle-duration', default=5.0, type=float,
              help='Maximum duration for a single subtitle segment (seconds)')
@click.option('--max-subtitle-chars', default=80, type=int,
              help='Maximum characters per subtitle line')
@click.option('--file-conflict', default='ask',
              type=click.Choice(['ask', 'overwrite', 'rename', 'skip']),
              help='How to handle existing output files')
@click.option('--hf-token', default=None,
              help='HuggingFace access token for speaker diarization models')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_path, model, language, output_format, embed, device, batch_size,
         compute_type, disable_vad, enable_diarization, enable_gender_classification,
         max_subtitle_duration, max_subtitle_chars, file_conflict, hf_token, verbose):
    """
    GPU-Optimized Bulk Video Transcription Tool

    Transcribe single video files or entire directories using optimized Whisper models.

    INPUT_PATH can be either:
    - A single video file (MP4, AVI, MKV, MOV, etc.)
    - A directory containing video files

    Examples:

        # Transcribe single file
        python main.py video.mp4

        # Transcribe all videos in directory
        python main.py /path/to/videos/ --model large-v3 --format srt

        # Embed subtitles into videos
        python main.py /path/to/videos/ --embed --format srt

        # High-speed processing with smaller model
        python main.py videos/ --model medium --batch-size 24
    """

    # Setup logging first
    logger = setup_logging(verbose)

    # Check basic dependencies first
    missing_deps = check_dependencies()
    if missing_deps:
        click.echo("❌ Missing required dependencies:")
        for dep in missing_deps:
            click.echo(f"   {dep}")
        sys.exit(1)

    # Import optional dependencies based on what's requested
    try:
        imports = import_optional_dependencies(enable_diarization, enable_gender_classification, hf_token)
    except SystemExit:
        # Re-raise sys.exit from dependency checking
        raise

    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            click.echo(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**2
            click.echo(f"📊 GPU Memory: {gpu_memory}MB")
        else:
            device = 'cpu'
            click.echo("⚠️  No CUDA GPU available, using CPU (will be slower)")

    # System info
    click.echo(f"🔧 CPU Cores: {psutil.cpu_count()}")
    click.echo(f"💾 RAM: {psutil.virtual_memory().total // 1024**3}GB")

    # Create configuration
    config = TranscriptionConfig(
        model_size=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        output_format=output_format,
        embed_subtitles=embed,
        enable_diarization=enable_diarization,
        enable_gender_classification=enable_gender_classification,
        max_subtitle_duration=max_subtitle_duration,
        max_subtitle_chars=max_subtitle_chars,
        file_conflict_mode=FileConflictMode(file_conflict)
    )

    # Add VAD setting to config
    config.disable_vad = disable_vad

    # Note: Gender classification can work with or without diarization
    if enable_gender_classification and not enable_diarization:
        click.echo("ℹ️  Note: Gender classification without diarization will classify the entire audio as one speaker")

    if enable_gender_classification and not imports.get('librosa') and not imports.get('speechbrain'):
        click.echo("❌ Warning: Gender classification enabled but no required libraries found")
        click.echo("Install: pip install librosa numpy")
        click.echo("Or: pip install speechbrain")
        click.echo("Gender classification will be disabled")

    try:
        # Initialize transcriber
        click.echo("🤖 Initializing transcription engine...")
        transcriber = BulkTranscriber(config, imports, hf_token)

        # Process videos
        input_path = pathlib.Path(input_path).resolve()
        click.echo(f"📂 Processing: {input_path}")

        results = transcriber.process_videos(input_path)

        # Display final results
        click.echo("\n" + "="*60)
        click.echo("🎉 TRANSCRIPTION COMPLETE")
        click.echo("="*60)
        click.echo(f"✅ Successfully processed: {results['processed']} videos")
        click.echo(f"❌ Failed: {results['failed']} videos")
        click.echo(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        click.echo(f"📈 Average time per video: {results['avg_time_per_video']:.2f} seconds")

        # Show created files
        if results.get('created_files'):
            click.echo(f"\n📁 OUTPUT FILES CREATED ({len(results['created_files'])}):")
            click.echo("-"*60)
            for file_path in results['created_files']:
                if pathlib.Path(file_path).exists():
                    file_size = pathlib.Path(file_path).stat().st_size
                    click.echo(f"  📄 {file_path}")
                    click.echo(f"      Size: {file_size:,} bytes")
                else:
                    click.echo(f"  ❌ {file_path} (NOT FOUND)")

        # Show errors summary
        if results.get('all_errors'):
            click.echo(f"\n⚠️  ERRORS SUMMARY ({len(results['all_errors'])}):")
            click.echo("-"*60)
            for error in results['all_errors'][:5]:  # Show first 5 errors
                click.echo(f"  ❌ {error}")
            if len(results['all_errors']) > 5:
                click.echo(f"  ... and {len(results['all_errors']) - 5} more errors")
            click.echo(f"  📄 Check transcription_detailed.log for full details")

        if results['processed'] > 0:
            click.echo(f"\n🎯 Transcriptions saved as .{output_format} files alongside original videos")
            if embed:
                click.echo("🎬 Subtitles embedded in video files (where applicable)")
            click.echo("📄 Detailed log saved to: transcription_detailed.log")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        click.echo(f"❌ Fatal Error: {e}", err=True)
        click.echo("📄 Check transcription_detailed.log for full error details", err=True)
        sys.exit(1)

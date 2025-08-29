import sys
import click
import torch
import psutil
import pathlib
import logging
import os
import warnings

# Set environment variables early to prevent Windows symlink permission issues
# This must be done before any SpeechBrain imports occur
os.environ['HF_HUB_CACHE_STRATEGY'] = 'copy'
os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'copy'

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

def _check_faster_whisper_cuda_support() -> bool:
    """Check if faster-whisper installation includes CUDA support."""
    try:
        # Method 1: Try to import CTranslate2 and check for CUDA
        try:
            import ctranslate2
            cuda_available = ctranslate2.get_cuda_device_count() > 0
            if cuda_available:
                return True
        except (ImportError, AttributeError):
            pass
        
        # Method 2: Check if faster-whisper can create a CUDA model
        try:
            from faster_whisper import WhisperModel
            # Try to instantiate a small model on CUDA
            test_model = WhisperModel("tiny", device="cuda", compute_type="float16")
            # If we get here without exception, CUDA support is available
            del test_model
            return True
        except Exception:
            pass
        
        # Method 3: Check package metadata for CUDA extras
        try:
            import pkg_resources
            import importlib.metadata
            
            # Check if installed with [cuda] extra
            try:
                dist = importlib.metadata.distribution('faster-whisper')
                if dist:
                    # This is a heuristic - the presence of ctranslate2 dependency suggests CUDA variant
                    requires = dist.requires or []
                    for req in requires:
                        if 'ctranslate2' in req and 'cuda' in req.lower():
                            return True
            except Exception:
                pass
            
            # Fallback: check if ctranslate2 is installed (usually means CUDA variant)
            try:
                import ctranslate2
                return True
            except ImportError:
                return False
                
        except ImportError:
            # No metadata available, assume CPU-only
            return False
        
        return False
        
    except Exception:
        # If we can't determine, assume no CUDA support
        return False

def import_optional_dependencies(enable_diarization: bool = False, enable_gender: bool = False, hf_token: str | None = None):
    """Import optional dependencies only when needed."""
    imports = {}

    # Transcription engines
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        imports['faster_whisper'] = (WhisperModel, BatchedInferencePipeline)
        
        # Check if faster-whisper has CUDA support
        cuda_support = _check_faster_whisper_cuda_support()
        imports['faster_whisper_cuda'] = cuda_support
        
    except ImportError:
        imports['faster_whisper'] = None
        imports['faster_whisper_cuda'] = False

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
    
    # Report faster-whisper variant
    if imports.get('faster_whisper'):
        cuda_support = imports.get('faster_whisper_cuda', False)
        if cuda_support:
            click.echo("📦 faster-whisper: CUDA variant detected ✅")
        else:
            click.echo("📦 faster-whisper: CPU-only variant detected")
            if device == 'cuda':
                click.echo("⚠️  Warning: GPU requested but faster-whisper lacks CUDA support")
                click.echo("💡 Install CUDA version: pip install faster-whisper[cuda]")
    elif imports.get('openai_whisper'):
        click.echo("📦 Using OpenAI Whisper")

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
        
        # Report final GPU usage status
        if config.device == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(0)
                if allocated > 0:
                    click.echo(f"✅ GPU is being used: {allocated // 1024**2}MB allocated")
                else:
                    click.echo("⚠️  Warning: GPU selected but no memory allocated")
                    click.echo("💡 Possible solutions:")
                    click.echo("   • Install CUDA-enabled faster-whisper: pip install faster-whisper[cuda]")
                    click.echo("   • Check CUDA installation: nvidia-smi")
                    click.echo("   • Try smaller model first: --model medium")
            except Exception:
                click.echo("⚠️  Warning: Could not verify GPU usage")
        else:
            click.echo("ℹ️  Using CPU mode")

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

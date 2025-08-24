# GPU-Optimized Bulk Video Transcription Tool

This is a high-performance Python application for transcribing video files in bulk. It leverages `faster-whisper` for GPU-optimized transcription and supports advanced features like speaker diarization and gender classification.

## Features

- **High-Speed Transcription**: Utilizes `faster-whisper` and CTranslate2 for optimal performance on NVIDIA GPUs (with float16/int8 precision).
- **Bulk Processing**: Transcribe a single video file or an entire directory of videos.
- **Speaker Diarization**: Automatically identify and label different speakers in the audio.
- **Gender Classification**: Classify speakers as MALE or FEMALE.
- **Multiple Output Formats**: Generate transcripts in SRT, VTT, TXT, or JSON formats.
- **Intelligent Subtitle Splitting**: Automatically breaks long transcription segments into shorter, more readable subtitles.
- **Subtitle Embedding**: Option to embed the generated subtitles as a soft-track into the original video file.
- **Flexible File Handling**: Choose how to handle existing transcript files (overwrite, skip, or rename).

## Prerequisites

Before you begin, ensure you have the following:

1.  **FFmpeg**: The script requires `ffmpeg` to extract audio from video files. Ensure it is installed on your system and accessible in your system's PATH.
    -   You can download it from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).

2.  **Hugging Face Account (for Speaker Diarization)**: The speaker diarization feature uses a gated model from Hugging Face. To use it, you must:
    -   **A. Have a Hugging Face Account:** If you don't have one, create one at [https://huggingface.co/join](https://huggingface.co/join).
    -   **B. Accept Model Terms:** Visit the page for the diarization model and accept the terms of service. You must be logged in.
        -   Model URL: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    -   **C. Get an Access Token:** Create an access token to authenticate with the Hugging Face Hub.
        -   Token URL: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

You will need to provide this token to the script using the `--hf-token` option.

## Installation

1.  **Clone the repository or download the source code.**

2.  **Install Dependencies**: It is recommended to use a virtual environment.

    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install required packages from the application directory
    pip install -r requirements.txt
    ```

## Usage

The application is run from the command line. You must run it as a module from the root directory of the project (`random_scripts`).

```bash
python -m gpu_transcriber_app.main <INPUT_PATH> [OPTIONS]
```

-   `<INPUT_PATH>`: The path to a single video file or a directory containing video files.

### Examples

**Transcribe a single file:**
```bash
python -m gpu_transcriber_app.main "C:\path\to\my\video.mp4"
```

**Transcribe all videos in a directory with diarization and gender classification:**
```bash
python -m gpu_transcriber_app.main "/path/to/videos/" --enable-diarization --enable-gender-classification --hf-token YOUR_HF_TOKEN
```

**Transcribe and embed subtitles into the video files:**
```bash
python -m gpu_transcriber_app.main "/path/to/videos/" --embed
```

### Command-Line Options

| Option | Alias | Description |
|---|---|---|
| `--model` | `-m` | Whisper model to use. Larger models are more accurate but slower. (Default: `large-v3`) |
| `--language` | `-l` | Force a specific language for transcription (e.g., "en", "es"). |
| `--output-format` | `-f` | The output format for the transcript file. (Default: `srt`) |
| `--embed` / `--no-embed` | | Embed subtitles into video files as a soft subtitle stream. |
| `--device` | `-d` | Processing device (`auto`, `cuda`, `cpu`). (Default: `auto`) |
| `--batch-size` | `-b` | Batch size for GPU processing. (Default: 16) |
| `--compute-type` | | Compute precision (`float16`, `float32`, `int8`). (Default: `float16`) |
| `--disable-vad` / `--enable-vad` | | Disable/Enable Voice Activity Detection. Disabling can be useful for music or mixed content. |
| `--enable-diarization` | | Enable speaker diarization to identify different speakers. **Requires Hugging Face setup.** |
| `--enable-gender-classification` | | Enable gender classification. Works with or without diarization. |
| `--max-subtitle-duration` | | Maximum duration for a single subtitle segment in seconds. (Default: 5.0) |
| `--max-subtitle-chars` | | Maximum characters per subtitle line. (Default: 80) |
| `--file-conflict` | | How to handle existing output files (`ask`, `overwrite`, `rename`, `skip`). (Default: `ask`) |
| `--hf-token` | | Your HuggingFace access token. **Required for speaker diarization.** See Prerequisites. |
| `--verbose` | `-v` | Enable verbose logging for debugging purposes. |
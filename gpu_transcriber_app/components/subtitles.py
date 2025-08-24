import json
import logging
import pathlib
from typing import List, Dict, Optional

import ffmpeg

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    """Generates subtitle files in various formats."""

    @staticmethod
    def break_long_segments(segments: List[Dict], max_duration: float = 5.0,
                           max_chars: int = 80) -> List[Dict]:
        """Intelligently break long segments based on natural speech patterns."""
        broken_segments = []

        # Sentence ending punctuation
        sentence_endings = {'.', '!', '?', '...', '。', '！', '？'}
        # Natural pause indicators
        pause_words = {'and', 'but', 'or', 'so', 'then', 'well', 'now', 'okay', 'right', 'um', 'uh'}

        for segment in segments:
            duration = segment['end'] - segment['start']
            text = segment['text'].strip()

            # If segment is reasonable length, keep it intact
            if duration <= max_duration * 1.4 and len(text) <= max_chars * 1.2:  # Allow 40% tolerance
                broken_segments.append(segment)
                continue

            # Only break if really necessary
            if duration <= max_duration * 2.0 and len(text) <= max_chars * 2.0:
                # For moderate overruns, try to find natural break points
                natural_break = SubtitleGenerator._find_natural_break_point(text, max_chars)
                if natural_break:
                    # Split at natural break point
                    first_part = text[:natural_break].strip()
                    second_part = text[natural_break:].strip()

                    if first_part and second_part:
                        # Estimate timing split
                        split_ratio = len(first_part) / len(text)
                        split_time = segment['start'] + duration * split_ratio

                        broken_segments.extend([
                            {
                                'start': segment['start'],
                                'end': split_time,
                                'text': first_part,
                                'speaker': segment.get('speaker', None),
                                'words': []
                            },
                            {
                                'start': split_time,
                                'end': segment['end'],
                                'text': second_part,
                                'speaker': segment.get('speaker', None),
                                'words': []
                            }
                        ])
                        continue

            # For very long segments, use word-level timestamps for precise breaking
            if segment.get('words') and len(segment['words']) > 1:
                broken_segments.extend(
                    SubtitleGenerator._break_with_word_timing(segment, max_duration, max_chars)
                )
            else:
                # Fallback: intelligent text-based splitting
                broken_segments.extend(
                    SubtitleGenerator._break_by_text_analysis(segment, max_duration, max_chars)
                )

        return broken_segments

    @staticmethod
    def _find_natural_break_point(text: str, max_chars: int) -> Optional[int]:
        """Find a natural place to break text (sentence boundary, comma, etc.)."""
        if len(text) <= max_chars:
            return None

        # Look for sentence endings first (within reasonable range)
        search_start = max_chars // 2
        search_end = int(min(len(text), max_chars * 1.2))

        for i in range(search_end, search_start, -1):
            if i < len(text) and text[i-1] in {'.', '!', '?'} and text[i] == ' ':
                return i

        # Look for comma + space
        for i in range(min(max_chars, len(text)), search_start, -1):
            if i < len(text) and text[i-1] == ',' and text[i] == ' ':
                return i

        # Look for conjunction words
        words = text.split()
        char_count = 0
        for i, word in enumerate(words):
            char_count += len(word) + (1 if i > 0 else 0)
            if char_count > search_start and word.lower() in {'and', 'but', 'or', 'so', 'then'}:
                # Return position after this word
                return char_count + 1 if char_count + 1 < len(text) else None

        return None

    @staticmethod
    def _break_with_word_timing(segment: Dict, max_duration: float, max_chars: int) -> List[Dict]:
        """Break segment using precise word-level timestamps."""
        words = segment.get('words', [])
        broken_segments = []
        current_text = ""
        current_start = segment['start']
        current_words = []

        for i, word in enumerate(words):
            word_text = word.get('word', '').strip()
            if not word_text:
                continue

            test_text = (current_text + " " + word_text).strip()
            word_end = word.get('end', current_start)
            current_duration = word_end - current_start

            # Check if we should break
            should_break = False

            if current_duration > max_duration * 1.5 or len(test_text) > max_chars * 1.5:
                # Definitely need to break
                should_break = True
            elif (current_duration > max_duration or len(test_text) > max_chars) and current_text:
                # Look for natural break point
                if (word_text.lower() in {'and', 'but', 'or', 'so', 'then', 'well', 'now'} or
                    (i > 0 and words[i-1].get('word', '').endswith(('.', '!', '?', ',')))):
                    should_break = True

            if should_break and current_text:
                # Save current segment
                broken_segments.append({
                    'start': current_start,
                    'end': words[i-1].get('end', current_start) if i > 0 else word.get('start', current_start),
                    'text': current_text,
                    'speaker': segment.get('speaker', None),
                    'words': current_words
                })
                current_text = word_text
                current_start = word.get('start', current_start)
                current_words = [word]
            else:
                current_text = test_text
                current_words.append(word)

        # Add final segment
        if current_text:
            broken_segments.append({
                'start': current_start,
                'end': segment['end'],
                'text': current_text,
                'speaker': segment.get('speaker', None),
                'words': current_words
            })

        return broken_segments

    @staticmethod
    def _break_by_text_analysis(segment: Dict, max_duration: float, max_chars: int) -> List[Dict]:
        """Break segment by analyzing text patterns when word timing is unavailable."""
        text = segment['text'].strip()
        duration = segment['end'] - segment['start']

        # Try to split at natural boundaries
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in {'.', '!', '?'} and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        if len(sentences) <= 1:
            # No natural sentences, split by estimated word timing
            words = text.split()
            target_words = max(1, int(len(words) * max_duration / duration))

            broken_segments = []
            for i in range(0, len(words), target_words):
                chunk_words = words[i:i + target_words]
                chunk_text = " ".join(chunk_words)

                chunk_start = segment['start'] + (duration * i / len(words))
                chunk_end = segment['start'] + (duration * (i + len(chunk_words)) / len(words))

                broken_segments.append({
                    'start': chunk_start,
                    'end': min(chunk_end, segment['end']),
                    'text': chunk_text,
                    'speaker': segment.get('speaker', None),
                    'words': []
                })

            return broken_segments

        # Group sentences intelligently
        broken_segments = []
        current_group = ""
        group_start = segment['start']
        chars_processed = 0

        for sentence in sentences:
            test_group = (current_group + " " + sentence).strip()
            sentence_ratio = len(sentence) / len(text)

            if (len(test_group) <= max_chars * 1.2 and
                len(current_group) > 0 and len(test_group) <= max_chars):
                current_group = test_group
            else:
                if current_group:
                    # Calculate timing for current group
                    group_ratio = chars_processed / len(text)
                    group_end = segment['start'] + duration * (chars_processed + len(current_group)) / len(text)

                    broken_segments.append({
                        'start': group_start,
                        'end': group_end,
                        'text': current_group,
                        'speaker': segment.get('speaker', None),
                        'words': []
                    })

                    chars_processed += len(current_group)
                    group_start = group_end

                current_group = sentence

        # Add final group
        if current_group:
            broken_segments.append({
                'start': group_start,
                'end': segment['end'],
                'text': current_group,
                'speaker': segment.get('speaker', None),
                'words': []
            })

        return broken_segments

    @staticmethod
    def assign_speakers(transcription_segments: List[Dict],
                       diarization_segments: List[Dict]) -> List[Dict]:
        """Assign speakers to transcription segments based on diarization."""
        if not diarization_segments:
            return transcription_segments

        def find_speaker(start_time: float, end_time: float) -> Optional[str]:
            """Find the most likely speaker for a given time range."""
            best_speaker = None
            max_overlap = 0

            for spk_seg in diarization_segments:
                # Calculate overlap between transcription segment and speaker segment
                overlap_start = max(start_time, spk_seg['start'])
                overlap_end = min(end_time, spk_seg['end'])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk_seg['speaker']

            return best_speaker

        # Assign speakers to each segment
        for segment in transcription_segments:
            speaker = find_speaker(segment['start'], segment['end'])
            segment['speaker'] = speaker

        return transcription_segments

    @staticmethod
    def apply_gender_labels(transcription_segments: List[Dict],
                           speaker_genders: Dict[str, str]) -> List[Dict]:
        """Apply gender labels to transcription segments."""
        for segment in transcription_segments:
            speaker = segment.get('speaker')
            if speaker and speaker in speaker_genders:
                segment['gender'] = speaker_genders[speaker]
            else:
                segment['gender'] = segment.get('gender', None)

        return transcription_segments

    @staticmethod
    def format_timestamp(seconds: float, format_type: str = "srt") -> str:
        """Format timestamp for different subtitle formats."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
        elif format_type == "vtt":
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @staticmethod
    def generate_srt(segments: List[Dict]) -> str:
        """Generate SRT format subtitle."""
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = SubtitleGenerator.format_timestamp(segment['start'], "srt")
            end_time = SubtitleGenerator.format_timestamp(segment['end'], "srt")
            text = segment['text'].strip()

            if text:  # Only add non-empty segments
                # Add gender label if available, otherwise use speaker label
                label = None
                if segment.get('gender') and segment.get('gender') != 'UNKNOWN':
                    label = segment['gender']
                elif segment.get('speaker'):
                    label = segment['speaker']

                if label:
                    text = f"[{label}] {text}"

                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

        return '\n'.join(srt_content)

    @staticmethod
    def generate_vtt(segments: List[Dict]) -> str:
        """Generate WebVTT format subtitle."""
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start_time = SubtitleGenerator.format_timestamp(segment['start'], "vtt")
            end_time = SubtitleGenerator.format_timestamp(segment['end'], "vtt")
            text = segment['text'].strip()

            if text:
                vtt_content.append(f"{start_time} --> {end_time}")
                vtt_content.append(f"{text}")
                vtt_content.append("")

        return '\n'.join(vtt_content)

    @staticmethod
    def generate_text(segments: List[Dict]) -> str:
        """Generate plain text transcript."""
        return '\n'.join(segment['text'].strip() for segment in segments if segment['text'].strip())

    @staticmethod
    def generate_json(segments: List[Dict]) -> str:
        """Generate JSON format with detailed timing."""
        return json.dumps(segments, indent=2, ensure_ascii=False)

class SubtitleEmbedder:
    """Embeds subtitles into video files."""

    @staticmethod
    def embed_soft_subtitles(video_path: pathlib.Path, subtitle_path: pathlib.Path,
                           output_path: pathlib.Path, language: str = "en") -> bool:
        """Embed subtitles as a soft subtitle stream."""
        try:
            # Convert paths to strings to handle network drives
            video_str = str(video_path)
            subtitle_str = str(subtitle_path)
            output_str = str(output_path)

            # Determine subtitle codec based on container
            container = output_path.suffix.lower()
            if container == '.mp4':
                subtitle_codec = 'mov_text'
            elif container == '.mkv':
                subtitle_codec = 'srt'
            else:
                subtitle_codec = 'srt'

            (
                ffmpeg
                .output(
                    ffmpeg.input(video_str),
                    ffmpeg.input(subtitle_str),
                    output_str,
                    **{'c': 'copy', f'c:s': subtitle_codec},
                    **{f'metadata:s:s:0': f'language={language}'}
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return True

        except ffmpeg.Error as e:
            logger.error(f"Subtitle embedding failed: {e.stderr.decode()}")
            return False

    @staticmethod
    def embed_hard_subtitles(video_path: pathlib.Path, subtitle_path: pathlib.Path,
                           output_path: pathlib.Path) -> bool:
        """Burn subtitles directly into video."""
        try:
            # Fix for network paths and f-string backslash issue
            subtitle_path_escaped = str(subtitle_path).replace(':', '\\:')

            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    vf=f"subtitles='{subtitle_path_escaped}'"
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return True

        except ffmpeg.Error as e:
            logger.error(f"Hard subtitle embedding failed: {e.stderr.decode()}")
            return False

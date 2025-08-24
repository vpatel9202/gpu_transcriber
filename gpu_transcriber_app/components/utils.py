
import logging
import sys
import pathlib
import time
import threading
from datetime import datetime
from typing import Tuple, Optional

import torch

from ..config import TranscriptionConfig, FileConflictMode

def setup_logging(verbose: bool = False):
    """Setup logging with appropriate level and format."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with detailed logging
    file_handler = logging.FileHandler('transcription_detailed.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)

    # Setup logger
    logger = logging.getLogger("gpu_transcriber_app")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class GPUMemoryManager:
    """Manages GPU memory for optimal RTX 4090 performance."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0

    def _get_gpu_memory(self) -> int:
        """Get total GPU memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // 1024**2
        return 0

    def get_optimal_batch_size(self, model_size: str) -> int:
        """Calculate optimal batch size based on GPU memory and model size."""
        memory_requirements = {
            "tiny": 1000,
            "base": 1000,
            "small": 2000,
            "medium": 5000,
            "large": 6000,
            "large-v3": 6000,
            "turbo": 5000
        }

        model_memory = memory_requirements.get(model_size, 6000)
        if self.total_memory > 15000:  # RTX 4090 (16GB+)
            return min(24, max(1, (self.total_memory - model_memory - 2000) // 500))
        elif self.total_memory > 10000:  # RTX 3080+ (12GB+)
            return min(16, max(1, (self.total_memory - model_memory - 1500) // 400))
        else:
            return min(8, max(1, (self.total_memory - model_memory - 1000) // 300))

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class ProgressTracker:
    """Tracks and displays real-time progress for long operations."""

    def __init__(self, logger):
        self.logger = logger
        self.current_operation = "Idle"
        self.last_update = time.time()
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.progress_thread = None

    def start_operation(self, operation_name: str):
        """Start tracking a new operation."""
        self.current_operation = operation_name
        self.start_time = time.time()
        self.last_update = time.time()
        self.stop_event.clear()

        # Start progress display thread
        self.progress_thread = threading.Thread(target=self._display_progress, daemon=True)
        self.progress_thread.start()

        self.logger.info(f"⏱️  Started: {operation_name}")

    def update_progress(self, message: str | None = None):
        """Update progress with optional message."""
        self.last_update = time.time()
        if message:
            self.logger.info(f"   → {message}")

    def finish_operation(self, success: bool = True):
        """Finish the current operation."""
        self.stop_event.set()
        if self.progress_thread:
            self.progress_thread.join(timeout=1.0)

        duration = time.time() - self.start_time
        status = "✅ Completed" if success else "❌ Failed"
        self.logger.info(f"{status}: {self.current_operation} ({duration:.1f}s)")
        self.current_operation = "Idle"

    def _display_progress(self):
        """Display progress indicators for long operations."""
        dots = 0
        while not self.stop_event.is_set():
            if time.time() - self.last_update > 30:  # No update for 30 seconds
                elapsed = time.time() - self.start_time
                dots = (dots + 1) % 4
                dot_display = "." * dots + " " * (3 - dots)
                print(f"\r🔄 {self.current_operation}{dot_display} ({elapsed:.0f}s)", end="", flush=True)

            if self.stop_event.wait(5):  # Check every 5 seconds
                break

        # Clear the progress line
        print("\r" + " " * 80 + "\r", end="", flush=True)

class FileHandler:
    """Handles file conflicts and user interactions."""

    def __init__(self, config: 'TranscriptionConfig'):
        self.config = config
        self.global_choice = None  # For "apply to all" functionality

    def handle_existing_file(self, file_path: pathlib.Path) -> Tuple[bool, Optional[pathlib.Path]]:
        """
        Handle existing file conflicts.
        Returns: (should_proceed, final_path)
        """
        if not file_path.exists():
            return True, file_path

        # Use global choice if set
        if self.global_choice:
            return self._apply_choice(self.global_choice, file_path)

        # Use configured mode
        if self.config.file_conflict_mode == FileConflictMode.OVERWRITE:
            return True, file_path
        elif self.config.file_conflict_mode == FileConflictMode.SKIP:
            return False, None
        elif self.config.file_conflict_mode == FileConflictMode.RENAME:
            return True, self._get_unique_filename(file_path)
        else:  # ASK mode
            return self._ask_user(file_path)

    def _ask_user(self, file_path: pathlib.Path) -> Tuple[bool, Optional[pathlib.Path]]:
        """Ask user what to do with existing file."""
        print(f"\n⚠️  File already exists: {file_path}")
        print("Choose an action:")
        print("  [o] Overwrite")
        print("  [r] Rename (add timestamp)")
        print("  [s] Skip this file")
        print("  [O] Overwrite ALL (apply to remaining files)")
        print("  [R] Rename ALL")
        print("  [S] Skip ALL")

        while True:
            try:
                choice = input("Enter choice [o/r/s/O/R/S]: ").strip().lower()

                if choice in ['o', 'overwrite']:
                    return True, file_path
                elif choice in ['r', 'rename']:
                    return True, self._get_unique_filename(file_path)
                elif choice in ['s', 'skip']:
                    return False, None
                elif choice in ['O']:
                    self.global_choice = 'overwrite'
                    return True, file_path
                elif choice in ['R']:
                    self.global_choice = 'rename'
                    return True, self._get_unique_filename(file_path)
                elif choice in ['S']:
                    self.global_choice = 'skip'
                    return False, None
                else:
                    print("Invalid choice. Please enter o, r, s, O, R, or S.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                sys.exit(1)

    def _apply_choice(self, choice: str, file_path: pathlib.Path) -> Tuple[bool, Optional[pathlib.Path]]:
        """Apply a saved global choice."""
        if choice == 'overwrite':
            return True, file_path
        elif choice == 'rename':
            return True, self._get_unique_filename(file_path)
        elif choice == 'skip':
            return False, None
        else:
            return True, file_path

    def _get_unique_filename(self, file_path: pathlib.Path) -> pathlib.Path:
        """Generate a unique filename by adding timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent

        new_name = f"{stem}_{timestamp}{suffix}"
        new_path = parent / new_name

        # If somehow this still exists, add a counter
        counter = 1
        while new_path.exists():
            new_name = f"{stem}_{timestamp}_{counter}{suffix}"
            new_path = parent / new_name
            counter += 1

        return new_path

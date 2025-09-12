"""
Audio Recording Manager for Interview Platform
Captures and manages audio streams from both interviewer and interviewee for cloud deployment.
"""

import io
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import logging

try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    import numpy as np
except ImportError as e:
    print(f"AudioRecordingManager: Missing required dependency: {e}")
    print("Please install dependencies: pip install pydub numpy")
    raise


class AudioRecordingManager:
    """
    Manages in-memory audio recording for complete interview sessions.
    Designed for cloud deployment with internal audio storage.
    """
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2):
        """
        Initialize the audio recording manager.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
            sample_width: Sample width in bytes (2 = 16-bit)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        
        # Audio storage
        self.audio_segments: List[Tuple[float, bytes, str]] = []  # (timestamp, audio_data, role)
        self.recording_lock = threading.Lock()
        self.session_start_time = None
        self.is_recording_session = False
        
        # Logger
        self.logger = self._setup_logger()
        
        self.logger.info(f"AudioRecordingManager initialized: {sample_rate}Hz, {channels}ch, {sample_width * 8}-bit")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the audio recording manager."""
        logger = logging.getLogger("AudioRecordingManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - AudioRecording: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_session(self):
        """Start a new recording session."""
        with self.recording_lock:
            self.session_start_time = time.time()
            self.is_recording_session = True
            self.audio_segments.clear()
            self.logger.info("Audio recording session started")
    
    def stop_session(self):
        """Stop the current recording session."""
        with self.recording_lock:
            self.is_recording_session = False
            self.logger.info(f"Audio recording session stopped. Captured {len(self.audio_segments)} segments")
    
    def add_interviewer_audio(self, audio_data: bytes):
        """
        Add interviewer audio segment to the recording.
        
        Args:
            audio_data: Raw audio bytes from Deepgram
        """
        if not self.is_recording_session:
            return
        
        current_time = time.time()
        with self.recording_lock:
            self.audio_segments.append((current_time, audio_data, "interviewer"))
            self.logger.debug(f"Added interviewer audio: {len(audio_data)} bytes at {current_time}")
    
    def add_interviewee_audio(self, audio_frames: List[bytes]):
        """
        Add interviewee audio segment to the recording.
        
        Args:
            audio_frames: List of audio frame bytes from recording
        """
        if not self.is_recording_session or not audio_frames:
            return
        
        # Combine frames into single audio data
        audio_data = b''.join(audio_frames)
        current_time = time.time()
        
        with self.recording_lock:
            self.audio_segments.append((current_time, audio_data, "interviewee"))
            self.logger.debug(f"Added interviewee audio: {len(audio_data)} bytes at {current_time}")
    
    def _create_audio_segment_from_bytes(self, audio_data: bytes) -> AudioSegment:
        """
        Create AudioSegment from raw audio bytes.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            AudioSegment object
        """
        try:
            # Create AudioSegment from raw bytes
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=self.sample_width,
                frame_rate=self.sample_rate,
                channels=self.channels
            )
            return audio_segment
        except Exception as e:
            self.logger.error(f"Failed to create audio segment: {e}")
            # Return empty audio segment as fallback
            return AudioSegment.silent(duration=0)
    
    def _sort_audio_segments_by_time(self) -> List[Tuple[float, bytes, str]]:
        """
        Sort audio segments by timestamp for chronological playback.
        
        Returns:
            Sorted list of audio segments
        """
        with self.recording_lock:
            return sorted(self.audio_segments, key=lambda x: x[0])
    
    def export_to_mp3(self, output_path: str) -> bool:
        """
        Export the complete interview audio to MP3 file.
        
        Args:
            output_path: Path where to save the MP3 file
            
        Returns:
            bool: Success status
        """
        try:
            if not self.audio_segments:
                self.logger.warning("No audio segments to export")
                return False
            
            self.logger.info(f"Exporting {len(self.audio_segments)} audio segments to MP3...")
            
            # Sort segments by timestamp
            sorted_segments = self._sort_audio_segments_by_time()
            
            # Create combined audio
            combined_audio = AudioSegment.empty()
            last_end_time = self.session_start_time if self.session_start_time else sorted_segments[0][0]
            
            for timestamp, audio_data, role in sorted_segments:
                # Create audio segment from bytes
                segment = self._create_audio_segment_from_bytes(audio_data)
                
                if len(segment) == 0:
                    continue
                
                # Calculate gap between segments and add silence if needed
                time_gap = timestamp - last_end_time
                if time_gap > 0.1:  # Add silence for gaps > 100ms
                    silence_duration = min(time_gap * 1000, 2000)  # Max 2 seconds silence
                    silence = AudioSegment.silent(duration=int(silence_duration))
                    combined_audio += silence
                    self.logger.debug(f"Added {silence_duration}ms silence gap")
                
                # Add the audio segment
                combined_audio += segment
                last_end_time = timestamp + (len(segment) / 1000.0)  # segment length in seconds
                
                self.logger.debug(f"Added {role} audio: {len(segment)}ms at {timestamp}")
            
            # Ensure we have some audio to export
            if len(combined_audio) == 0:
                self.logger.warning("Combined audio is empty")
                return False
            
            # Export to MP3
            self.logger.info(f"Exporting combined audio ({len(combined_audio)}ms) to {output_path}")
            combined_audio.export(output_path, format="mp3", bitrate="128k")
            
            self.logger.info(f"Successfully exported interview audio to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export MP3: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_to_wav(self, output_path: str) -> bool:
        """
        Export the complete interview audio to WAV file (backup format).
        
        Args:
            output_path: Path where to save the WAV file
            
        Returns:
            bool: Success status
        """
        try:
            if not self.audio_segments:
                self.logger.warning("No audio segments to export")
                return False
            
            self.logger.info(f"Exporting {len(self.audio_segments)} audio segments to WAV...")
            
            # Sort segments by timestamp
            sorted_segments = self._sort_audio_segments_by_time()
            
            # Prepare WAV file
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                
                last_end_time = self.session_start_time if self.session_start_time else sorted_segments[0][0]
                
                for timestamp, audio_data, role in sorted_segments:
                    # Calculate gap and add silence if needed
                    time_gap = timestamp - last_end_time
                    if time_gap > 0.1:  # Add silence for gaps > 100ms
                        silence_duration = min(time_gap, 2.0)  # Max 2 seconds silence
                        silence_frames = int(self.sample_rate * silence_duration)
                        silence_data = b'\x00' * (silence_frames * self.sample_width * self.channels)
                        wav_file.writeframes(silence_data)
                        self.logger.debug(f"Added {silence_duration}s silence gap")
                    
                    # Write audio data
                    wav_file.writeframes(audio_data)
                    last_end_time = timestamp + (len(audio_data) / (self.sample_rate * self.sample_width * self.channels))
                    
                    self.logger.debug(f"Added {role} audio: {len(audio_data)} bytes at {timestamp}")
            
            self.logger.info(f"Successfully exported interview audio to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export WAV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_session_stats(self) -> dict:
        """
        Get statistics about the current recording session.
        
        Returns:
            Dictionary with session statistics
        """
        with self.recording_lock:
            total_segments = len(self.audio_segments)
            interviewer_segments = sum(1 for _, _, role in self.audio_segments if role == "interviewer")
            interviewee_segments = sum(1 for _, _, role in self.audio_segments if role == "interviewee")
            
            total_audio_bytes = sum(len(audio_data) for _, audio_data, _ in self.audio_segments)
            estimated_duration = total_audio_bytes / (self.sample_rate * self.sample_width * self.channels)
            
            session_duration = (time.time() - self.session_start_time) if self.session_start_time else 0
            
            return {
                "total_segments": total_segments,
                "interviewer_segments": interviewer_segments,
                "interviewee_segments": interviewee_segments,
                "total_audio_bytes": total_audio_bytes,
                "estimated_duration_seconds": estimated_duration,
                "session_duration_seconds": session_duration,
                "is_recording": self.is_recording_session
            }
    
    def clear_session(self):
        """Clear all recorded audio from memory."""
        with self.recording_lock:
            self.audio_segments.clear()
            self.session_start_time = None
            self.is_recording_session = False
            self.logger.info("Audio recording session cleared")
    
    def create_timestamped_filename(self, base_dir: str, prefix: str = "interview_recording") -> str:
        """
        Create a timestamped filename for the recording.
        
        Args:
            base_dir: Base directory for the file
            prefix: Filename prefix
            
        Returns:
            Full path with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_{timestamp}.mp3"
        return str(Path(base_dir) / filename)

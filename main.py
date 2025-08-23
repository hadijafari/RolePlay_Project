"""
AI Interview Platform - Audio Recording System
Spacebar-controlled audio recording with timestamp-based file naming.
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

try:
    import pyaudio
    import keyboard
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies: pip install pyaudio keyboard")
    sys.exit(1)

from config.settings import AudioConfig, STTConfig
from services.stt_service import STTService
from services.tts_service import TTSService


class AudioRecorder:
    """Handles audio recording with spacebar control and STT integration."""
    
    def __init__(self, stt_service=None, tts_service=None):
        self.audio = pyaudio.PyAudio()
        self.config = AudioConfig()
        self.stt_config = STTConfig()
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.is_recording = False
        self.recording_thread = None
        self.transcription_thread = None
        self.frames = []
        self.current_filename = None
        self.current_filepath = None
        
        # Ensure directories exist
        os.makedirs(self.config.RECORDINGS_DIR, exist_ok=True)
        if self.stt_config.SAVE_TRANSCRIPTS:
            os.makedirs(self.stt_config.TRANSCRIPTS_DIR, exist_ok=True)
        
    def check_microphone_access(self):
        """Test microphone access and display available devices."""
        try:
            print("Interview Platform: Checking microphone access...")
            
            # List available audio devices
            print("\nAvailable audio devices:")
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"  Device {i}: {device_info['name']} (Input channels: {device_info['maxInputChannels']})")
            
            # Test opening audio stream
            test_stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK,
                input_device_index=None  # Use default device
            )
            test_stream.close()
            print("✓ Microphone access confirmed")
            return True
            
        except Exception as e:
            print(f"✗ Microphone access error: {e}")
            print("Please check:")
            print("  - Microphone is connected and enabled")
            print("  - Microphone permissions are granted")
            print("  - No other application is using the microphone")
            return False
    
    def generate_filename(self):
        """Generate timestamp-based filename for recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"interview_recording_{timestamp}.{self.config.OUTPUT_FORMAT}"
    
    def start_recording(self):
        """Start audio recording in a separate thread."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.frames = []
        self.current_filename = self.generate_filename()
        
        print(f"Interview Platform: Recording started - {self.current_filename}")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop audio recording, save file, and optionally transcribe."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()
        
        # Save the recording
        if self.frames:
            self._save_recording()
            print(f"Interview Platform: Recording saved - {self.current_filename}")
            
            # Start transcription if STT is enabled
            if (self.stt_service and 
                self.stt_config.ENABLED and 
                self.stt_config.AUTO_TRANSCRIBE and 
                self.current_filepath):
                self._start_transcription()
        else:
            print("Interview Platform: No audio data recorded")
    
    def _record_audio(self):
        """Internal method to record audio data."""
        try:
            stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            
            while self.is_recording:
                data = stream.read(self.config.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Interview Platform: Recording error - {e}")
            self.is_recording = False
    
    def _save_recording(self):
        """Save recorded audio frames to MP3 file for faster processing."""
        try:
            self.current_filepath = os.path.join(self.config.RECORDINGS_DIR, self.current_filename)
            
            # Convert raw audio data to MP3 using pydub
            try:
                from pydub import AudioSegment
                from pydub.utils import make_chunks
                
                # Create audio segment from raw PCM data
                audio_data = b''.join(self.frames)
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=self.audio.get_sample_size(self.config.FORMAT),
                    frame_rate=self.config.RATE,
                    channels=self.config.CHANNELS
                )
                
                # Export as MP3 with optimized settings
                audio_segment.export(
                    self.current_filepath,
                    format=self.config.OUTPUT_FORMAT,
                    bitrate=self.config.MP3_BITRATE,
                    parameters=["-q:a", str(self.config.MP3_QUALITY)]
                )
                
                # Display file info
                file_size = os.path.getsize(self.current_filepath)
                duration = len(self.frames) * self.config.CHUNK / self.config.RATE
                
                # Calculate compression ratio (compared to WAV)
                wav_size = len(self.frames) * self.config.CHUNK * self.audio.get_sample_size(self.config.FORMAT) * self.config.CHANNELS
                compression_ratio = wav_size / file_size if file_size > 0 else 1
                
                print(f"Interview Platform: File size: {file_size:,} bytes, Duration: {duration:.1f}s")
                print(f"Interview Platform: MP3 compression ratio: {compression_ratio:.1f}x smaller than WAV")
                print(f"Interview Platform: Expected latency improvement: {min(50, (1 - 1/compression_ratio) * 100):.0f}% faster uploads")
                
            except ImportError:
                print("Interview Platform: Warning - pydub not available, falling back to WAV format")
                # Fallback to WAV if pydub is not available
                import wave
                with wave.open(self.current_filepath.replace('.mp3', '.wav'), 'wb') as wf:
                    wf.setnchannels(self.config.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.config.FORMAT))
                    wf.setframerate(self.config.RATE)
                    wf.writeframes(b''.join(self.frames))
                self.current_filepath = self.current_filepath.replace('.mp3', '.wav')
                self.current_filename = self.current_filename.replace('.mp3', '.wav')
                
                file_size = os.path.getsize(self.current_filepath)
                duration = len(self.frames) * self.config.CHUNK / self.config.RATE
                print(f"Interview Platform: WAV fallback - File size: {file_size:,} bytes, Duration: {duration:.1f}s")
            
        except Exception as e:
            print(f"Interview Platform: Error saving recording - {e}")
            self.current_filepath = None
    
    def _start_transcription(self):
        """Start transcription in a separate thread."""
        if self.transcription_thread and self.transcription_thread.is_alive():
            print("Interview Platform: Previous transcription still in progress")
            return
        
        print("Interview Platform: Starting transcription...")
        self.transcription_thread = threading.Thread(target=self._transcribe_audio)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def _transcribe_audio(self):
        """Internal method to transcribe recorded audio."""
        try:
            if not self.current_filepath or not os.path.exists(self.current_filepath):
                print("Interview Platform: No valid audio file for transcription")
                return
            
            # Perform transcription
            result = self.stt_service.transcribe_audio(self.current_filepath)
            
            if result["success"]:
                transcription_text = result["text"]
                processing_time = result["processing_time"]
                
                print(f"Interview Platform: Transcription completed in {processing_time:.2f}s")
                print(f"Interview Platform: Transcribed text: \"{transcription_text}\"")
                
                # Voice feedback - repeat what was said
                if self.tts_service:
                    self._voice_feedback(transcription_text)
                
                # Save transcript if enabled
                if self.stt_config.SAVE_TRANSCRIPTS:
                    self._save_transcript(transcription_text, result)
                    
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"Interview Platform: Transcription failed - {error_msg}")
                
        except Exception as e:
            print(f"Interview Platform: Transcription error - {e}")
    
    def _save_transcript(self, text: str, result: dict):
        """Save transcription to file."""
        try:
            # Generate transcript filename based on audio filename
            audio_name = Path(self.current_filename).stem
            timestamp = result.get("timestamp", datetime.now().isoformat())
            
            if self.stt_config.TRANSCRIPT_FORMAT == "json":
                import json
                transcript_filename = f"{audio_name}_transcript.json"
                transcript_path = os.path.join(self.stt_config.TRANSCRIPTS_DIR, transcript_filename)
                
                transcript_data = {
                    "text": text,
                    "audio_file": self.current_filename,
                    "timestamp": timestamp,
                    "processing_time": result.get("processing_time", 0),
                    "language": result.get("language", "unknown"),
                    "model": result.get("model", "whisper-1")
                }
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                    
            else:  # txt format
                transcript_filename = f"{audio_name}_transcript.txt"
                transcript_path = os.path.join(self.stt_config.TRANSCRIPTS_DIR, transcript_filename)
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(f"Audio File: {self.current_filename}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
                    f.write(f"Language: {result.get('language', 'unknown')}\n")
                    f.write(f"Model: {result.get('model', 'whisper-1')}\n")
                    f.write("-" * 50 + "\n")
                    f.write(text)
            
            print(f"Interview Platform: Transcript saved - {transcript_filename}")
            
        except Exception as e:
            print(f"Interview Platform: Error saving transcript - {e}")
    
    def _voice_feedback(self, text: str):
        """Use TTS to repeat the transcribed text."""
        try:
            if not self.tts_service:
                return
                
            print("Interview Platform: Voice feedback - repeating what you said...")
            
            # Generate and play speech
            result = self.tts_service.generate_speech(text, play_immediately=True)
            
            if result["success"]:
                print(f"Interview Platform: Voice feedback completed in {result['processing_time']:.2f}s")
                print(f"Interview Platform: Voice used: {result['voice']}")
            else:
                print(f"Interview Platform: Voice feedback failed - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Interview Platform: Voice feedback error - {e}")
    
    def cleanup(self):
        """Clean up PyAudio resources."""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()
        
        # Clean up TTS service
        if self.tts_service:
            try:
                self.tts_service.cleanup()
            except Exception as e:
                print(f"Interview Platform: TTS cleanup error - {e}")


class InterviewPlatform:
    """Main application class for the AI Interview Platform."""
    
    def __init__(self):
        # Initialize STT service if enabled
        self.stt_service = None
        stt_config = STTConfig()
        
        if stt_config.ENABLED:
            try:
                self.stt_service = STTService()
                print("Interview Platform: STT service initialized")
            except Exception as e:
                print(f"Interview Platform: STT service initialization failed - {e}")
                print("Interview Platform: Continuing without STT functionality")
        
        # Initialize TTS service for voice feedback
        self.tts_service = None
        try:
            self.tts_service = TTSService()
            print("Interview Platform: TTS service initialized")
        except Exception as e:
            print(f"Interview Platform: TTS service initialization failed - {e}")
            print("Interview Platform: Continuing without TTS functionality")
        
        self.recorder = AudioRecorder(stt_service=self.stt_service, tts_service=self.tts_service)
        self.running = False
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        print("=" * 60)
        print("🎤 AI Interview Platform - Audio Recording System")
        if self.stt_service:
            print("    with Speech-to-Text Transcription")
        if self.tts_service:
            print("    with Voice Feedback (repeats what you say)")
        print("=" * 60)
        print("\nControls:")
        print("  SPACEBAR (hold) : Start/Stop recording")
        print("  ESC or Ctrl+C   : Exit application")
        print("\nRecordings will be saved in the 'recordings/' directory")
        print("with timestamp-based filenames.")
        
        if self.stt_service:
            stt_config = STTConfig()
            print(f"\nSTT Features:")
            print(f"  Auto-transcribe : {'Enabled' if stt_config.AUTO_TRANSCRIBE else 'Disabled'}")
            print(f"  Save transcripts: {'Enabled' if stt_config.SAVE_TRANSCRIPTS else 'Disabled'}")
            print(f"  Model           : {stt_config.MODEL}")
            print(f"  Target latency  : {stt_config.TARGET_LATENCY}s")
        
        print("\nPress ESC to exit or hold SPACEBAR to start recording...")
        print("-" * 60)
    
    def setup_keyboard_handlers(self):
        """Set up keyboard event handlers."""
        # Spacebar press/release handlers
        keyboard.on_press_key('space', self._on_spacebar_press)
        keyboard.on_release_key('space', self._on_spacebar_release)
        
        # Escape key handler
        keyboard.on_press_key('esc', self._on_escape_press)
    
    def _on_spacebar_press(self, event):
        """Handle spacebar press event."""
        if not self.recorder.is_recording:
            self.recorder.start_recording()
    
    def _on_spacebar_release(self, event):
        """Handle spacebar release event."""
        if self.recorder.is_recording:
            self.recorder.stop_recording()
    
    def _on_escape_press(self, event):
        """Handle escape key press event."""
        print("\nInterview Platform: Shutting down...")
        self.running = False
    
    def run(self):
        """Main application loop."""
        try:
            # Check microphone access
            if not self.recorder.check_microphone_access():
                return
            
            # Test STT service connection if enabled
            if self.stt_service:
                print("Interview Platform: Testing STT service connection...")
                if self.stt_service.test_connection():
                    print("Interview Platform: STT service ready")
                else:
                    print("Interview Platform: STT service connection failed")
                    print("Interview Platform: Continuing with recording only")
                    self.stt_service = None
                    self.recorder.stt_service = None
            
            # Test TTS service connection if enabled
            if self.tts_service:
                print("Interview Platform: Testing TTS service connection...")
                if self.tts_service.test_connection():
                    print("Interview Platform: TTS service ready")
                else:
                    print("Interview Platform: TTS service connection failed")
                    print("Interview Platform: Continuing without voice feedback")
                    self.tts_service = None
                    self.recorder.tts_service = None
            
            # Display welcome message
            self.display_welcome()
            
            # Set up keyboard handlers
            self.setup_keyboard_handlers()
            
            # Main loop
            self.running = True
            while self.running:
                time.sleep(0.1)  # Small delay to prevent high CPU usage
            
        except KeyboardInterrupt:
            print("\nInterview Platform: Interrupted by user")
        
        except Exception as e:
            print(f"Interview Platform: Unexpected error - {e}")
        
        finally:
            # Cleanup
            self.recorder.cleanup()
            
            # Display final statistics if STT was used
            if self.stt_service:
                stats = self.stt_service.get_stats()
                if stats["total_requests"] > 0:
                    print(f"\nInterview Platform: STT Statistics:")
                    print(f"  Total requests: {stats['total_requests']}")
                    print(f"  Success rate: {stats['success_rate']:.1%}")
                    print(f"  Average processing time: {stats['average_processing_time']:.2f}s")
            
            # Display final statistics if TTS was used
            if self.tts_service:
                stats = self.tts_service.get_stats()
                if stats["total_requests"] > 0:
                    print(f"\nInterview Platform: TTS Statistics:")
                    print(f"  Total requests: {stats['total_requests']}")
                    print(f"  Success rate: {stats['success_rate']:.1%}")
                    print(f"  Average processing time: {stats['average_processing_time']:.2f}s")
                    print(f"  Cache hits: {stats['cache_hits']}")
                    print(f"  Cache misses: {stats['cache_misses']}")
            
            print("Interview Platform: Goodbye!")


def main():
    """Entry point for the application."""
    try:
        app = InterviewPlatform()
        app.run()
    except Exception as e:
        print(f"Interview Platform: Failed to start - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

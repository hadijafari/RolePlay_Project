"""
Configuration settings for the AI Interview Platform audio recording system.
"""

import os
import pyaudio
from pathlib import Path


class AudioConfig:
    """Audio recording configuration settings."""
    
    # Audio format settings
    FORMAT = pyaudio.paInt16  # 16-bit audio
    CHANNELS = 1              # Mono audio
    RATE = 16000             # Sample rate (16 kHz - optimized for speech)
    CHUNK = 1024             # Audio buffer size
    
    # Output format settings
    OUTPUT_FORMAT = "mp3"     # MP3 for smaller file sizes and faster uploads
    MP3_BITRATE = "128k"     # 128 kbps - good quality for speech, small file size
    MP3_QUALITY = 3          # Quality setting (0=best, 9=worst) - 3 is good balance
    
    # File settings
    BASE_DIR = Path(__file__).parent.parent
    RECORDINGS_DIR = BASE_DIR / "recordings"
    
    # Recording settings
    MAX_RECORDING_DURATION = 3600  # Maximum recording duration in seconds (1 hour)
    
    @classmethod
    def get_audio_info(cls):
        """Get formatted audio configuration information."""
        return {
            "format": "16-bit PCM",
            "channels": cls.CHANNELS,
            "sample_rate": f"{cls.RATE} Hz",
            "buffer_size": cls.CHUNK,
            "max_duration": f"{cls.MAX_RECORDING_DURATION} seconds"
        }
    
    @classmethod
    def validate_settings(cls):
        """Validate audio settings and environment."""
        issues = []
        
        # Check if recordings directory exists or can be created
        try:
            os.makedirs(cls.RECORDINGS_DIR, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create recordings directory: {e}")
        
        # Check audio settings
        if cls.RATE <= 0:
            issues.append("Sample rate must be positive")
        
        if cls.CHANNELS <= 0:
            issues.append("Number of channels must be positive")
        
        if cls.CHUNK <= 0:
            issues.append("Chunk size must be positive")
        
        return issues


class AppConfig:
    """General application configuration settings."""
    
    # Application metadata
    APP_NAME = "AI Interview Platform"
    VERSION = "1.0.0"
    AUTHOR = "AI Interview System"
    
    # UI settings
    WELCOME_MESSAGE = """
🎤 AI Interview Platform - Audio Recording System

This platform provides spacebar-controlled audio recording for interview sessions.
All recordings are automatically saved with timestamps for easy organization.
"""
    
    # Keyboard shortcuts
    RECORD_KEY = "space"
    EXIT_KEY = "esc"
    
    # File naming
    FILENAME_PREFIX = "interview_recording"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    @classmethod
    def get_app_info(cls):
        """Get formatted application information."""
        return {
            "name": cls.APP_NAME,
            "version": cls.VERSION,
            "author": cls.AUTHOR,
            "record_key": cls.RECORD_KEY.upper(),
            "exit_key": cls.EXIT_KEY.upper()
        }


class SystemConfig:
    """System-specific configuration settings."""
    
    # Platform detection
    import platform
    PLATFORM = platform.system().lower()
    
    # Thread settings
    MAIN_LOOP_DELAY = 0.1  # Seconds between main loop iterations
    
    # Error handling
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # Seconds between retry attempts
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_system_info(cls):
        """Get formatted system information."""
        return {
            "platform": cls.PLATFORM,
            "main_loop_delay": f"{cls.MAIN_LOOP_DELAY}s",
            "max_retries": cls.MAX_RETRY_ATTEMPTS,
            "log_level": cls.LOG_LEVEL
        }


class STTConfig:
    """Speech-to-Text configuration settings."""
    
    # OpenAI Whisper API settings
    MODEL = "whisper-1"
    ENABLED = True  # Enable/disable STT functionality
    
    # Processing settings
    AUTO_TRANSCRIBE = True  # Automatically transcribe after recording
    SAVE_TRANSCRIPTS = True  # Save transcripts to files
    TRANSCRIPT_FORMAT = "txt"  # Format for transcript files (txt, json)
    
    # Performance settings
    TARGET_LATENCY = 2.0  # Target processing time in seconds
    MAX_RETRY_ATTEMPTS = 3
    TIMEOUT = 30.0  # API request timeout
    
    # File settings
    BASE_DIR = Path(__file__).parent.parent
    TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
    
    # Language settings
    DEFAULT_LANGUAGE = None  # Auto-detect language
    SUPPORTED_LANGUAGES = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'
    ]
    
    @classmethod
    def get_stt_info(cls):
        """Get formatted STT configuration information."""
        return {
            "model": cls.MODEL,
            "enabled": cls.ENABLED,
            "auto_transcribe": cls.AUTO_TRANSCRIBE,
            "target_latency": f"{cls.TARGET_LATENCY}s",
            "max_retries": cls.MAX_RETRY_ATTEMPTS,
            "timeout": f"{cls.TIMEOUT}s",
            "transcript_format": cls.TRANSCRIPT_FORMAT
        }
    
    @classmethod
    def validate_stt_settings(cls):
        """Validate STT settings and environment."""
        issues = []
        
        # Check if transcripts directory exists or can be created
        if cls.SAVE_TRANSCRIPTS:
            try:
                os.makedirs(cls.TRANSCRIPTS_DIR, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create transcripts directory: {e}")
        
        # Check API key availability
        import os
        if cls.ENABLED and not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not found in environment variables")
        
        # Check settings validity
        if cls.TARGET_LATENCY <= 0:
            issues.append("Target latency must be positive")
        
        if cls.MAX_RETRY_ATTEMPTS <= 0:
            issues.append("Max retry attempts must be positive")
        
        if cls.TIMEOUT <= 0:
            issues.append("Timeout must be positive")
        
        return issues


# Export main configuration classes
__all__ = ['AudioConfig', 'AppConfig', 'SystemConfig', 'STTConfig']

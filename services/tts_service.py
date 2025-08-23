"""
Text-to-Speech Service using ElevenLabs API
Handles text-to-audio conversion with voice customization and audio playback.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save
    import pygame
    from dotenv import load_dotenv
except ImportError as e:
    print(f"TTS Service: Missing required dependency: {e}")
    print("Please install dependencies: pip install elevenlabs pygame python-dotenv")
    sys.exit(1)

# Load environment variables
# Navigate to the agents directory where .env is located
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class TTSConfig:
    """Configuration for Text-to-Speech service."""
    
    # ElevenLabs API settings
    API_KEY_ENV = "ELEVENLABS_MAIN_API_KEY"
    MAX_TEXT_LENGTH = 2500  # Maximum characters per API call
    CHUNK_OVERLAP = 100     # Characters to overlap between chunks
    
    # Voice settings
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - professional female
    DEFAULT_MODEL = "eleven_monolingual_v1"     # High-quality model
    
    # Audio playback settings
    DEFAULT_VOLUME = 0.8
    DEFAULT_SPEED = 1.0
    DEFAULT_CLARITY = 0.75
    
    # Performance settings
    TARGET_LATENCY = 3.0    # Target TTS generation time in seconds
    TIMEOUT = 60.0          # API request timeout in seconds
    MAX_RETRY_ATTEMPTS = 3
    
    # File settings
    BASE_DIR = Path(__file__).parent.parent
    TTS_CACHE_DIR = BASE_DIR / "tts_cache"
    AUDIO_FORMAT = "mp3"
    
    # Voice customization ranges
    SPEED_RANGE = (0.5, 2.0)      # 0.5x to 2.0x speed
    CLARITY_RANGE = (0.0, 1.0)    # 0.0 to 1.0 clarity
    VOLUME_RANGE = (0.1, 1.0)     # 0.1 to 1.0 volume


class VoiceProfile:
    """Represents a voice profile with customization options."""
    
    def __init__(self, voice_id: str, name: str, description: str = "", 
                 gender: str = "neutral", accent: str = "neutral", 
                 professional_level: str = "medium"):
        self.voice_id = voice_id
        self.name = name
        self.description = description
        self.gender = gender
        self.accent = accent
        self.professional_level = professional_level
        
        # Default customization settings
        self.speed = 2.0                # default  = 1.0
        self.clarity = 0.75             # default  = 0.75
        self.volume = 0.8               # default  = 0.8
        
    def __str__(self):
        return f"{self.name} ({self.gender}, {self.accent}, {self.professional_level})"
    
    def get_customization_info(self):
        """Get formatted customization information."""
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "description": self.description,
            "gender": self.gender,
            "accent": self.accent,
            "professional_level": self.professional_level,
            "speed": self.speed,
            "clarity": self.clarity,
            "volume": self.volume
        }


class TTSService:
    """Text-to-Speech service using ElevenLabs API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TTS service with ElevenLabs API key.
        
        Args:
            api_key: ElevenLabs API key. If None, loads from environment.
        """
        self.config = TTSConfig()
        self.logger = self._setup_logger()
        
        # Initialize ElevenLabs API
        self.api_key = api_key or os.getenv(self.config.API_KEY_ENV)
        if not self.api_key:
            raise ValueError("TTS Service: ElevenLabs API key not found. Please set ELEVENLABS_MAIN_API_KEY in .env file")
        
        self.client = ElevenLabs(api_key=self.api_key)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize voice profiles
        self.voice_profiles = self._initialize_voice_profiles()
        self.current_voice = self.voice_profiles[0]  # Default to first voice
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "total_audio_duration": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Ensure cache directory exists
        os.makedirs(self.config.TTS_CACHE_DIR, exist_ok=True)
        
        self.logger.info("TTS Service: Initialized with ElevenLabs API")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for TTS service."""
        logger = logging.getLogger("TTSService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - TTS Service: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_voice_profiles(self) -> List[VoiceProfile]:
        """Initialize available voice profiles."""
        profiles = [
            VoiceProfile(
                voice_id="21m00Tcm4TlvDq8ikWAM",
                name="Rachel",
                description="Professional female voice, clear and articulate",
                gender="female",
                accent="American",
                professional_level="high"
            ),
            VoiceProfile(
                voice_id="AZnzlk1XvdvUeBnXmlld",
                name="Domi",
                description="Professional male voice, warm and engaging",
                gender="male",
                accent="American",
                professional_level="high"
            ),
            VoiceProfile(
                voice_id="EXAVITQu4vr4xnSDxMaL",
                name="Bella",
                description="Friendly female voice, approachable and clear",
                gender="female",
                accent="British",
                professional_level="medium"
            ),
            VoiceProfile(
                voice_id="VR6AewLTigWG4xSOukaG",
                name="Arnold",
                description="Confident male voice, authoritative and clear",
                gender="male",
                accent="American",
                professional_level="high"
            ),
            VoiceProfile(
                voice_id="pNInz6obpgDQGcFmaJgB",
                name="Adam",
                description="Casual male voice, friendly and natural",
                gender="male",
                accent="American",
                professional_level="medium"
            )
        ]
        
        self.logger.info(f"TTS Service: Initialized {len(profiles)} voice profiles")
        return profiles
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voice profiles."""
        return [profile.get_customization_info() for profile in self.voice_profiles]
    
    def select_voice(self, voice_id: str) -> bool:
        """Select a voice profile by ID."""
        for profile in self.voice_profiles:
            if profile.voice_id == voice_id:
                self.current_voice = profile
                self.logger.info(f"TTS Service: Selected voice: {profile.name}")
                return True
        
        self.logger.warning(f"TTS Service: Voice ID {voice_id} not found")
        return False
    
    def set_voice_customization(self, speed: Optional[float] = None, 
                               clarity: Optional[float] = None, 
                               volume: Optional[float] = None) -> bool:
        """Set voice customization parameters."""
        try:
            if speed is not None:
                if self.config.SPEED_RANGE[0] <= speed <= self.config.SPEED_RANGE[1]:
                    self.current_voice.speed = speed
                else:
                    self.logger.warning(f"Speed {speed} out of range {self.config.SPEED_RANGE}")
                    return False
            
            if clarity is not None:
                if self.config.CLARITY_RANGE[0] <= clarity <= self.config.CLARITY_RANGE[1]:
                    self.current_voice.clarity = clarity
                else:
                    self.logger.warning(f"Clarity {clarity} out of range {self.config.CLARITY_RANGE}")
                    return False
            
            if volume is not None:
                if self.config.VOLUME_RANGE[0] <= volume <= self.config.VOLUME_RANGE[1]:
                    self.current_voice.volume = volume
                else:
                    self.logger.warning(f"Volume {volume} out of range {self.config.VOLUME_RANGE}")
                    return False
            
            self.logger.info(f"TTS Service: Voice customization updated - Speed: {self.current_voice.speed}, "
                           f"Clarity: {self.current_voice.clarity}, Volume: {self.current_voice.volume}")
            return True
            
        except Exception as e:
            self.logger.error(f"TTS Service: Error setting voice customization: {e}")
            return False
    
    def _generate_cache_key(self, text: str, voice_id: str, speed: float, clarity: float) -> str:
        """Generate a cache key for the TTS request."""
        import hashlib
        content = f"{text}_{voice_id}_{speed}_{clarity}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[str]:
        """Get cached audio file if it exists."""
        cache_file = self.config.TTS_CACHE_DIR / f"{cache_key}.{self.config.AUDIO_FORMAT}"
        if cache_file.exists():
            self.stats["cache_hits"] += 1
            self.logger.debug(f"TTS Service: Cache hit for key: {cache_key}")
            return str(cache_file)
        
        self.stats["cache_misses"] += 1
        return None
    
    def _save_to_cache(self, cache_key: str, audio_data: bytes) -> str:
        """Save generated audio to cache."""
        cache_file = self.config.TTS_CACHE_DIR / f"{cache_key}.{self.config.AUDIO_FORMAT}"
        with open(cache_file, 'wb') as f:
            f.write(audio_data)
        
        self.logger.debug(f"TTS Service: Cached audio for key: {cache_key}")
        return str(cache_file)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Break long text into manageable chunks."""
        if len(text) <= self.config.MAX_TEXT_LENGTH:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.MAX_TEXT_LENGTH
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.config.CHUNK_OVERLAP
            if start >= len(text):
                break
        
        self.logger.info(f"TTS Service: Split text into {len(chunks)} chunks")
        return chunks
    
    def generate_speech(self, text: str, play_immediately: bool = True) -> Dict[str, Any]:
        """
        Generate speech from text using ElevenLabs API.
        
        Args:
            text: Text to convert to speech
            play_immediately: Whether to play audio immediately after generation
            
        Returns:
            Dict containing generation result and metadata
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Generate cache key
            cache_key = self._generate_cache_key(
                text, 
                self.current_voice.voice_id, 
                self.current_voice.speed, 
                self.current_voice.clarity
            )
            
            # Check cache first
            cached_audio = self._get_cached_audio(cache_key)
            if cached_audio:
                audio_file_path = cached_audio
                processing_time = time.time() - start_time
                
                if play_immediately:
                    self._play_audio(audio_file_path)
                
                return {
                    "success": True,
                    "audio_file": audio_file_path,
                    "processing_time": processing_time,
                    "cached": True,
                    "voice": self.current_voice.name,
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Generate speech
            self.logger.info(f"TTS Service: Generating speech for text ({len(text)} characters)")
            
            # Split text into chunks if needed
            text_chunks = self._chunk_text(text)
            audio_files = []
            
            for i, chunk in enumerate(text_chunks):
                chunk_start = time.time()
                
                # Generate audio using ElevenLabs API
                audio_response = self.client.text_to_speech.convert(
                    text=chunk,
                    voice_id=self.current_voice.voice_id,
                    model_id=self.config.DEFAULT_MODEL
                )
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f'.{self.config.AUDIO_FORMAT}', 
                    delete=False
                )
                
                # Handle the response (it might be a generator or bytes)
                if hasattr(audio_response, '__iter__') and not isinstance(audio_response, bytes):
                    # It's a generator, collect all chunks
                    audio_data = b''.join(audio_response)
                else:
                    # It's already bytes
                    audio_data = audio_response
                
                with open(temp_file.name, 'wb') as f:
                    f.write(audio_data)
                temp_file.close()
                
                audio_files.append(temp_file.name)
                chunk_time = time.time() - chunk_start
                
                self.logger.debug(f"TTS Service: Generated chunk {i+1}/{len(text_chunks)} in {chunk_time:.2f}s")
            
            # Combine audio files if multiple chunks
            if len(audio_files) == 1:
                final_audio_path = audio_files[0]
            else:
                final_audio_path = self._combine_audio_chunks(audio_files)
            
            # Save to cache
            with open(final_audio_path, 'rb') as f:
                audio_data = f.read()
            self._save_to_cache(cache_key, audio_data)
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_requests"] += 1
            self._update_average_processing_time()
            
            # Play audio if requested
            if play_immediately:
                self._play_audio(final_audio_path)
            
            self.logger.info(f"TTS Service: Speech generation completed in {processing_time:.2f}s")
            
            return {
                "success": True,
                "audio_file": final_audio_path,
                "processing_time": processing_time,
                "cached": False,
                "voice": self.current_voice.name,
                "text_length": len(text),
                "chunks": len(text_chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            self.logger.error(f"TTS Service: Speech generation failed after {processing_time:.2f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _combine_audio_chunks(self, audio_files: List[str]) -> str:
        """Combine multiple audio chunks into a single file."""
        try:
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            for audio_file in audio_files:
                segment = AudioSegment.from_file(audio_file)
                combined += segment
            
            # Save combined audio
            combined_file = tempfile.NamedTemporaryFile(
                suffix=f'.{self.config.AUDIO_FORMAT}', 
                delete=False
            )
            combined.export(combined_file.name, format=self.config.AUDIO_FORMAT)
            combined_file.close()
            
            # Clean up individual chunk files
            for audio_file in audio_files:
                try:
                    os.unlink(audio_file)
                except:
                    pass
            
            return combined_file.name
            
        except Exception as e:
            self.logger.error(f"TTS Service: Error combining audio chunks: {e}")
            # Return first file if combination fails
            return audio_files[0]
    
    def _play_audio(self, audio_file_path: str):
        """Play audio file using pygame."""
        try:
            # Stop any currently playing audio
            pygame.mixer.music.stop()
            
            # Load and play the audio
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.set_volume(self.current_voice.volume)
            pygame.mixer.music.play()
            
            self.logger.info(f"TTS Service: Playing audio: {Path(audio_file_path).name}")
            
        except Exception as e:
            self.logger.error(f"TTS Service: Error playing audio: {e}")
    
    def stop_audio(self):
        """Stop currently playing audio."""
        try:
            pygame.mixer.music.stop()
            self.logger.info("TTS Service: Audio playback stopped")
        except Exception as e:
            self.logger.error(f"TTS Service: Error stopping audio: {e}")
    
    def pause_audio(self):
        """Pause currently playing audio."""
        try:
            pygame.mixer.music.pause()
            self.logger.info("TTS Service: Audio playback paused")
        except Exception as e:
            self.logger.error(f"TTS Service: Error pausing audio: {e}")
    
    def resume_audio(self):
        """Resume paused audio playback."""
        try:
            pygame.mixer.music.unpause()
            self.logger.info("TTS Service: Audio playback resumed")
        except Exception as e:
            self.logger.error(f"TTS Service: Error resuming audio: {e}")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return pygame.mixer.music.get_busy()
    
    def get_playback_position(self) -> float:
        """Get current playback position in seconds."""
        try:
            return pygame.mixer.music.get_pos() / 1000.0
        except:
            return 0.0
    
    def test_connection(self) -> bool:
        """Test connection to ElevenLabs API."""
        try:
            # Try to get available voices
            available_voices = self.client.voices.get_all()
            if available_voices:
                self.logger.info("TTS Service: API connection test successful")
                return True
            else:
                self.logger.warning("TTS Service: API connection test: No voices returned")
                return False
                
        except Exception as e:
            self.logger.error(f"TTS Service: API connection test failed: {e}")
            return False
    
    def _update_average_processing_time(self):
        """Update average processing time statistics."""
        if self.stats["successful_requests"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"] 
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "current_voice": self.current_voice.get_customization_info(),
            "config": {
                "max_text_length": self.config.MAX_TEXT_LENGTH,
                "target_latency": self.config.TARGET_LATENCY,
                "timeout": self.config.TIMEOUT,
                "max_retries": self.config.MAX_RETRY_ATTEMPTS
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            pygame.mixer.quit()
            self.logger.info("TTS Service: Cleanup completed")
        except Exception as e:
            self.logger.error(f"TTS Service: Error during cleanup: {e}")


# Export main classes
__all__ = ['TTSService', 'TTSConfig', 'VoiceProfile']

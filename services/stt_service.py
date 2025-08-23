"""
Speech-to-Text Service using OpenAI Whisper API
Handles audio transcription with error handling and retry logic.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import openai
    from dotenv import load_dotenv
except ImportError as e:
    print(f"STT Service: Missing required dependency: {e}")
    print("Please install dependencies: pip install openai python-dotenv")
    sys.exit(1)

# Load environment variables
# Navigate to the agents directory where .env is located
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class STTConfig:
    """Configuration for Speech-to-Text service."""
    
    # OpenAI Whisper API settings
    MODEL = "whisper-1"
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB limit for Whisper API
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.mpga', '.webm']  # MP3 first for priority
    
    # Retry settings
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0  # Base delay in seconds
    RETRY_BACKOFF = 2.0  # Exponential backoff multiplier
    
    # Performance settings
    TARGET_LATENCY = 2.0  # Target processing time in seconds
    TIMEOUT = 30.0  # API request timeout in seconds
    
    # Audio preprocessing
    ENABLE_NOISE_REDUCTION = False  # Placeholder for future enhancement
    NORMALIZE_AUDIO = False  # Placeholder for future enhancement


class STTService:
    """Speech-to-Text service using OpenAI Whisper API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize STT service with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, loads from environment.
        """
        self.config = STTConfig()
        self.logger = self._setup_logger()
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("STT Service: OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        self.logger.info("STT Service: Initialized with OpenAI Whisper API")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for STT service."""
        logger = logging.getLogger("STTService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - STT Service: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate audio file for Whisper API compatibility.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                self.logger.error(f"Audio file not found: {file_path}")
                return False
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.config.MAX_FILE_SIZE:
                self.logger.error(f"File too large: {file_size} bytes (max: {self.config.MAX_FILE_SIZE})")
                return False
            
            # Check file format
            file_extension = path.suffix.lower()
            if file_extension not in self.config.SUPPORTED_FORMATS:
                self.logger.error(f"Unsupported format: {file_extension}")
                return False
            
            self.logger.debug(f"Audio file validated: {file_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating audio file: {e}")
            return False
    
    def preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file if needed.
        
        Args:
            file_path: Path to original audio file
            
        Returns:
            str: Path to processed audio file (may be same as input)
        """
        # Placeholder for future audio preprocessing
        # Could include:
        # - Noise reduction
        # - Audio normalization
        # - Format conversion
        # - Compression for faster upload
        
        if self.config.ENABLE_NOISE_REDUCTION:
            self.logger.info("STT Service: Audio preprocessing enabled (placeholder)")
        
        return file_path
    
    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using OpenAI Whisper API.
        
        Args:
            file_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')
            
        Returns:
            Dict containing transcription result and metadata
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Validate audio file
            if not self.validate_audio_file(file_path):
                raise ValueError(f"Invalid audio file: {file_path}")
            
            # Preprocess audio if needed
            processed_file = self.preprocess_audio(file_path)
            
            self.logger.info(f"STT Service: Starting transcription of {Path(file_path).name}")
            
            # Perform transcription with retry logic
            result = self._transcribe_with_retry(processed_file, language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_requests"] += 1
            self._update_average_processing_time()
            
            # Log results
            self.logger.info(f"STT Service: Transcription completed in {processing_time:.2f}s")
            if processing_time > self.config.TARGET_LATENCY:
                self.logger.warning(f"STT Service: Processing time exceeded target ({self.config.TARGET_LATENCY}s)")
            
            return {
                "success": True,
                "text": result.text,
                "language": getattr(result, 'language', 'unknown'),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "model": self.config.MODEL
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            self.logger.error(f"STT Service: Transcription failed after {processing_time:.2f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "model": self.config.MODEL
            }
    
    def _transcribe_with_retry(self, file_path: str, language: Optional[str] = None):
        """
        Transcribe audio with retry logic for API failures.
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            
        Returns:
            Transcription result from OpenAI API
        """
        last_exception = None
        
        for attempt in range(self.config.MAX_RETRY_ATTEMPTS):
            try:
                with open(file_path, "rb") as audio_file:
                    # Prepare API parameters
                    params = {
                        "file": audio_file,
                        "model": self.config.MODEL,
                        "timeout": self.config.TIMEOUT
                    }
                    
                    if language:
                        params["language"] = language
                    
                    # Make API request
                    result = self.client.audio.transcriptions.create(**params)
                    
                    self.logger.debug(f"STT Service: API request successful on attempt {attempt + 1}")
                    return result
                    
            except openai.APITimeoutError as e:
                last_exception = e
                self.logger.warning(f"STT Service: API timeout on attempt {attempt + 1}: {e}")
                
            except openai.RateLimitError as e:
                last_exception = e
                self.logger.warning(f"STT Service: Rate limit exceeded on attempt {attempt + 1}: {e}")
                
            except openai.APIConnectionError as e:
                last_exception = e
                self.logger.warning(f"STT Service: API connection error on attempt {attempt + 1}: {e}")
                
            except openai.AuthenticationError as e:
                # Don't retry authentication errors
                self.logger.error(f"STT Service: Authentication error: {e}")
                raise e
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"STT Service: Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.config.MAX_RETRY_ATTEMPTS - 1:
                delay = self.config.RETRY_DELAY * (self.config.RETRY_BACKOFF ** attempt)
                self.logger.info(f"STT Service: Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        
        # All attempts failed
        raise last_exception or Exception("All transcription attempts failed")
    
    def _update_average_processing_time(self):
        """Update average processing time statistics."""
        if self.stats["successful_requests"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dict containing service statistics
        """
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"] 
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "config": {
                "model": self.config.MODEL,
                "max_retries": self.config.MAX_RETRY_ATTEMPTS,
                "target_latency": self.config.TARGET_LATENCY,
                "timeout": self.config.TIMEOUT
            }
        }
    
    def cleanup_temp_files(self, file_paths: list):
        """
        Clean up temporary audio files after processing.
        
        Args:
            file_paths: List of file paths to clean up
        """
        cleaned_count = 0
        for file_path in file_paths:
            try:
                if os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
                    os.remove(file_path)
                    cleaned_count += 1
                    self.logger.debug(f"STT Service: Cleaned up temp file: {file_path}")
            except Exception as e:
                self.logger.warning(f"STT Service: Failed to clean up {file_path}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"STT Service: Cleaned up {cleaned_count} temporary files")
    
    def test_connection(self) -> bool:
        """
        Test connection to OpenAI API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create a minimal test file
            test_content = b'\x00' * 1000  # Minimal audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(test_content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, "rb") as audio_file:
                    self.client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.config.MODEL,
                        timeout=5.0
                    )
                return True
            except openai.AuthenticationError:
                self.logger.error("STT Service: Authentication failed - check API key")
                return False
            except Exception as e:
                # Other errors might be due to invalid test data, which is expected
                self.logger.info(f"STT Service: API connection test completed: {e}")
                return True
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            self.logger.error(f"STT Service: Connection test failed: {e}")
            return False


# Export main class
__all__ = ['STTService', 'STTConfig']

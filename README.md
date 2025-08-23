# 🎤 AI Interview Platform

A Python-based audio recording system designed for interview sessions with spacebar-controlled recording functionality and OpenAI Whisper Speech-to-Text integration.

## Features

### Audio Recording
- **Spacebar Control**: Hold spacebar to record, release to stop
- **Timestamp Naming**: Automatic filename generation with timestamps
- **MP3 Format**: Optimized audio recording in MP3 format (16kHz, 128kbps, mono) for faster processing
- **Error Handling**: Robust microphone access and error management
- **Real-time Feedback**: Console messages for recording status

### Speech-to-Text (STT)
- **OpenAI Whisper Integration**: Automatic transcription using whisper-1 model
- **Auto-transcription**: Immediate processing after recording stops
- **Multi-format Support**: Saves transcripts as TXT or JSON files
- **Language Detection**: Automatic language identification
- **Error Recovery**: Retry logic for API failures with exponential backoff
- **Performance Monitoring**: Target latency under 2 seconds
- **Accuracy Optimization**: Handles various audio qualities and accents
- **Performance Optimization**: MP3 format with 16kHz sample rate for 30-50% faster processing

### Text-to-Speech (TTS)
- **ElevenLabs Integration**: High-quality voice synthesis using professional voice models
- **Multiple Voice Profiles**: 5 professional voices (Rachel, Domi, Bella, Arnold, Adam)
- **Voice Customization**: Adjustable speed (0.5x-2.0x), clarity (0.0-1.0), volume (0.1-1.0)
- **Audio Playback**: Real-time audio playback with pygame controls
- **Smart Chunking**: Automatic text splitting for long content
- **Caching System**: Efficient storage and retrieval of generated audio
- **Professional Voices**: Interview-appropriate voices with different accents and styles

### System Features
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Environment Management**: UV-compatible for dependency management
- **Configurable**: Extensive settings for audio and STT parameters

## Project Structure

```
interview_platform/
├── main.py              # Main application with recording and STT
├── test_stt.py          # STT functionality test suite
├── requirements.txt     # Python dependencies
├── config/
│   └── settings.py     # Audio and STT configuration settings
├── services/
│   └── stt_service.py  # OpenAI Whisper STT service
├── recordings/         # Directory for saved audio recordings
├── transcripts/        # Directory for saved transcriptions
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Working microphone
- Audio drivers installed
- OpenAI API key (for STT functionality)
- Internet connection (for STT API calls)

### Setup with UV (Recommended)

Since you're using UV for environment management:

1. **Navigate to the project directory:**
   ```bash
   cd "D:\workspaces\AI-Tutorials\AI Agents\MyAgentsTutorial\agents\2_openai\Interview RolePlay\interview_platform"
   ```

2. **Set up environment variables:**
   - Ensure the `.env` file in the parent directory contains your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install dependencies using UV:**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Test the STT functionality (optional):**
   ```bash
   uv run test_stt.py
   ```

5. **Test the TTS functionality (optional):**
   ```bash
   uv run test_tts.py
   ```

5. **Run the application:**
   ```bash
   uv run main.py
   ```

### Alternative Installation (Standard pip)

1. **Create virtual environment:**
   ```bash
   python -m venv interview_env
   ```

2. **Activate virtual environment:**
   - Windows: `interview_env\Scripts\activate`
   - macOS/Linux: `source interview_env/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Starting the Application

Run the main script:
```bash
uv run main.py  # Using UV
# or
python main.py  # Using standard Python
```

### Controls

| Key | Action |
|-----|--------|
| **SPACEBAR** (hold) | Start recording / Stop recording |
| **ESC** | Exit application |
| **Ctrl+C** | Force exit |

### Recording Process

1. **Start the application** - The system will check microphone access and STT connectivity
2. **Hold SPACEBAR** - Recording begins immediately with visual feedback
3. **Release SPACEBAR** - Recording stops, file is saved, and transcription begins automatically
4. **Wait for transcription** - STT processing happens in background (target: <2s)
5. **Check results**:
   - **recordings/** - Find your audio files with timestamps
   - **transcripts/** - Find corresponding transcription files

### File Naming Convention

**Audio Files:**
```
interview_recording_YYYYMMDD_HHMMSS.mp3
```
Example: `interview_recording_20250122_143052.mp3`

**Transcript Files:**
```
interview_recording_YYYYMMDD_HHMMSS_transcript.txt
interview_recording_YYYYMMDD_HHMMSS_transcript.json  # If JSON format enabled
```

## Configuration

### Audio Settings

Edit `config/settings.py` to customize audio recording:

```python
class AudioConfig:
    FORMAT = pyaudio.paInt16  # 16-bit audio
    CHANNELS = 1              # Mono audio
    RATE = 16000             # Sample rate (16 kHz - optimized for speech)
    CHUNK = 1024             # Buffer size
    OUTPUT_FORMAT = "mp3"     # MP3 for smaller file sizes
    MP3_BITRATE = "128k"     # 128 kbps - good quality, small size
    MP3_QUALITY = 3          # Quality setting (0=best, 9=worst)
    RECORDINGS_DIR = BASE_DIR / "recordings"
```

### STT Settings

Configure Speech-to-Text functionality:

```python
class STTConfig:
    MODEL = "whisper-1"           # OpenAI Whisper model
    ENABLED = True                # Enable/disable STT
    AUTO_TRANSCRIBE = True        # Auto-transcribe after recording
    SAVE_TRANSCRIPTS = True       # Save transcript files
    TRANSCRIPT_FORMAT = "txt"     # "txt" or "json"
    TARGET_LATENCY = 2.0         # Target processing time (seconds)
    MAX_RETRY_ATTEMPTS = 3       # API retry attempts
    TIMEOUT = 30.0               # API request timeout
    TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
```

### TTS Settings

Configure Text-to-Speech functionality:

```python
class TTSConfig:
    API_KEY_ENV = "ELEVENLABS_MAIN_API_KEY"  # Environment variable name
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - professional female
    DEFAULT_MODEL = "eleven_monolingual_v1"     # High-quality model
    TARGET_LATENCY = 3.0                       # Target TTS generation time
    MAX_TEXT_LENGTH = 2500                     # Max characters per API call
    SPEED_RANGE = (0.5, 2.0)                  # Speed customization range
    CLARITY_RANGE = (0.0, 1.0)                # Clarity customization range
    VOLUME_RANGE = (0.1, 1.0)                 # Volume customization range
```

## Troubleshooting

### Common Issues

**1. Microphone Access Denied**
- Check microphone permissions in system settings
- Ensure microphone is not being used by another application
- Try running as administrator (Windows)

**2. PyAudio Installation Issues**
- Windows: Install Microsoft Visual C++ Build Tools
- macOS: `brew install portaudio`
- Linux: `sudo apt-get install portaudio19-dev`

**3. Keyboard Module Issues**
- May require administrator/root privileges
- On Linux: Add user to input group

**4. No Audio Devices Found**
- Check that microphone is properly connected
- Update audio drivers
- Restart the application

**5. STT Service Errors**
- Verify OPENAI_API_KEY is correctly set in .env file
- Check internet connection
- Ensure OpenAI API credits are available
- Try running `uv run test_stt.py` to diagnose issues

**6. Transcription Failures**
- Check audio quality (clear speech, minimal background noise)
- Ensure recording duration is reasonable (not too short/long)
- Verify supported audio format (WAV files work best)

### Error Messages

| Error | Solution |
|-------|----------|
| "Missing required dependency" | Install requirements: `uv pip install -r requirements.txt` |
| "Microphone access error" | Check permissions and device availability |
| "No audio data recorded" | Ensure microphone is working and not muted |
| "STT service initialization failed" | Check OPENAI_API_KEY and internet connection |
| "Transcription failed" | Verify API key, credits, and audio quality |
| "API connection error" | Check internet connection and OpenAI service status |

## Technical Details

### Available TTS Voices

The platform includes 5 professional voice profiles optimized for interviews:

| Voice | Gender | Accent | Professional Level | Description |
|-------|--------|--------|-------------------|-------------|
| **Rachel** | Female | American | High | Professional female voice, clear and articulate |
| **Domi** | Male | American | High | Professional male voice, warm and engaging |
| **Bella** | Female | British | Medium | Friendly female voice, approachable and clear |
| **Arnold** | Male | American | High | Confident male voice, authoritative and clear |
| **Adam** | Male | American | Medium | Casual male voice, friendly and natural |

### Audio Specifications

- **Format**: MP3 (128 kbps, optimized for speech)
- **Channels**: Mono (1 channel)
- **Sample Rate**: 16 kHz (optimized for speech recognition)
- **Buffer Size**: 1024 frames
- **Max Duration**: 1 hour per recording
- **File Size**: ~1MB per minute (vs ~5MB for WAV)
- **Latency Improvement**: 30-50% faster uploads to OpenAI API

### Dependencies

**Core Audio:**
- **pyaudio**: Audio recording interface
- **keyboard**: Global keyboard event handling
- **pydub**: MP3 encoding and audio processing

**Speech-to-Text:**
- **openai**: OpenAI API client for Whisper STT
- **python-dotenv**: Environment variable management

**Text-to-Speech:**
- **elevenlabs**: ElevenLabs API client for voice synthesis
- **pygame**: Audio playback and control system

**Optional:**
- **numpy**: Audio data processing (for test suite)

### System Requirements

- **RAM**: Minimum 100MB available
- **Storage**: ~5MB per minute of recording
- **CPU**: Any modern processor
- **OS**: Windows 7+, macOS 10.12+, Linux (most distributions)

## Development

### Adding Features

The modular design allows easy extension:

1. **Custom Audio Formats**: Modify `AudioConfig` class
2. **Additional Controls**: Add keyboard handlers in `InterviewPlatform`
3. **File Processing**: Extend `AudioRecorder` class
4. **UI Enhancements**: Modify display methods

### Testing

Test microphone functionality:
```python
python -c "import pyaudio; p = pyaudio.PyAudio(); print('Audio system ready')"
```

## License

This project is part of the AI Agents Tutorial series.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Test microphone with other applications
4. Review console output for specific error messages

---

**Happy Recording! 🎤**

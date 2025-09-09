# Deepgram Voice Agent Integration Guide

## Overview
The Enhanced AI Interview Platform now supports real-time voice conversations using Deepgram's Voice Agent API. This replaces the previous three-step process (STT → GPT → TTS) with a single WebSocket connection for lower latency and more natural conversations.

## Key Features
- **TAB-controlled recording**: Hold TAB to record, release to send to Deepgram
- **Complete audio processing**: Sends complete audio chunks for better reliability
- **Low latency**: Single API handles transcription, response generation, and speech synthesis
- **Native conversation history**: Built-in history management through Deepgram API
- **Uses GPT-4o-mini**: Still leverages OpenAI's language model for responses
- **Preserves all document analysis features**: Resume and job description analysis work as before

## Prerequisites

### Required API Keys
Add the following to your `.env` file in `D:\workspaces\AI-Tutorials\AI Agents\MyAgentsTutorial\agents`:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Installation
```bash
# Install required dependencies
pip install deepgram-sdk>=3.0.0
# or
uv pip install deepgram-sdk>=3.0.0

# Install all requirements
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

## Usage

### Running the Interview Platform

```bash
# Navigate to the platform directory
cd "2_openai/Interview RolePlay/interview_platform"

# Run with Deepgram Voice Agent (default)
python main.py

# Or run with UV
uv run main.py
```

### How to Use TAB Recording

1. **Start the platform** - The system will connect to Deepgram
2. **Wait for the ready message** - You'll see "READY FOR INTERVIEW"
3. **Hold TAB key** - Start recording your response
4. **Speak your response** - While holding TAB
5. **Release TAB** - Your audio is sent to Deepgram
6. **Listen to response** - The agent's response will play automatically
7. **Repeat** - Hold TAB again for your next response

### Fallback to Original Mode
If you want to use the original STT/TTS mode:

```python
# In main.py, modify the initialization
platform = EnhancedInterviewPlatform(use_deepgram=False)
```

## How It Works

1. **Document Analysis** (Unchanged)
   - Place resume and job description in the RAG directory
   - System analyzes documents and creates interview plan
   - Interview conductor is initialized with context

2. **Voice Agent Connection**
   - Establishes WebSocket connection to Deepgram
   - Configures STT (Nova-3), LLM (GPT-4o-mini), and TTS (Aura-2)
   - Sets up TAB key handlers for recording

3. **TAB-Controlled Conversation**
   - Hold TAB to record your response
   - Release TAB to send audio to Deepgram
   - Deepgram transcribes, generates response, and synthesizes speech
   - Audio response plays through speakers automatically
   - Conversation history is maintained natively

4. **Session End**
   - Press Ctrl+C to end the interview
   - Conversation history is automatically saved
   - Connection closes gracefully

## Architecture Changes

### Before (Three-Step Process)
```
TAB Hold → PyAudio Recording → TAB Release → OpenAI Whisper STT → 
GPT-4o-mini → ElevenLabs TTS → Speaker Playback
```

### After (Single WebSocket with TAB Recording)
```
TAB Hold → Record Audio → TAB Release → Send to Deepgram Voice Agent → Speaker
                                                ↓
                                    [STT + GPT-4o-mini + TTS internally]
```

## Configuration

### Deepgram Settings (in `config/settings.py`)
```python
class DeepgramConfig:
    # Speech Recognition
    STT_MODEL = "nova-3"  # Deepgram's latest model
    
    # Language Model
    LLM_MODEL = "gpt-4o-mini"  # OpenAI model
    
    # Text-to-Speech
    TTS_MODEL = "aura-2-thalia-en"  # Professional voice
    
    # Audio Settings
    INPUT_SAMPLE_RATE = 16000  # Microphone rate
    OUTPUT_SAMPLE_RATE = 16000  # Playback rate
```

## Troubleshooting

### Connection Issues
- Verify DEEPGRAM_API_KEY is set correctly
- Check internet connection
- Ensure firewall allows WebSocket connections

### Audio Issues
- Test microphone with system settings
- Adjust volume levels
- Check PyAudio installation

### Fallback Mode
If Deepgram fails, the system automatically falls back to the original STT/TTS mode.

## Features Status

| Feature | Status |
|---------|---------|
| Document Analysis | ✅ Working (unchanged) |
| Interview Planning | ✅ Working (unchanged) |
| Interview Conductor | ✅ Working (adapted for TAB recording) |
| TAB Recording | ✅ Working (hold TAB to record) |
| Voice Conversation | ✅ Working (via Deepgram) |
| Conversation History | ✅ Native support |
| Feedback Agent | ⏸️ Disabled (for testing) |

## Next Steps

Once you've verified the Deepgram integration works:
1. Re-enable the feedback agent if needed
2. Fine-tune voice agent prompts
3. Add custom function calling for dynamic actions
4. Implement session recording/playback

## Support

For issues or questions:
- Check Deepgram documentation: https://developers.deepgram.com/docs/voice-agent
- Review error logs in the console
- Verify all API keys are valid

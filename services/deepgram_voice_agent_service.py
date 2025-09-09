"""
Deepgram Voice Agent Service for Real-time Interview Conversations
Replaces the STT->GPT->TTS pipeline with a single WebSocket connection
"""

import os
import sys
import time
import logging
import threading
import asyncio
import queue
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        AgentWebSocketEvents,
        AgentKeepAlive,
    )
    from deepgram.clients.agent.v1.websocket.options import SettingsOptions
    import pyaudio
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Deepgram Voice Agent Service: Missing required dependency: {e}")
    print("Please install dependencies: pip install deepgram-sdk pyaudio python-dotenv")
    sys.exit(1)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class DeepgramVoiceAgentService:
    """
    Service for managing real-time voice conversations using Deepgram Voice Agent API.
    
    This service provides:
    - Real-time speech-to-text transcription
    - GPT-4o-mini powered response generation  
    - Text-to-speech synthesis
    - Bidirectional audio streaming
    - Native conversation history management
    """
    
    def __init__(self, interview_conductor=None, interview_plan=None):
        """
        Initialize Deepgram Voice Agent service.
        
        Args:
            interview_conductor: Optional interview conductor agent for context
            interview_plan: Optional interview plan for structured interviews
        """
        self.logger = self._setup_logger()
        
        # API Configuration
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in .env file")
        
        # Interview components
        self.interview_conductor = interview_conductor
        self.interview_plan = interview_plan
        
        # Connection state
        self.connection = None
        self.is_connected = False
        self.is_recording = False
        self.keep_alive_thread = None
        
        # Audio configuration
        self.audio = pyaudio.PyAudio()
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.input_rate = 16000  # Microphone input rate
        self.chunk_size = 1024
        self.stream = None
        
        # Audio output handling
        self.audio_output_queue = queue.Queue()
        self.audio_buffer = bytearray()
        self.output_stream = None
        
        # Conversation tracking
        self.conversation_log = []
        self.current_turn_count = 0
        self.processing_audio = False
        
        # Statistics
        self.stats = {
            "total_turns": 0,
            "connection_time": 0,
            "total_audio_sent": 0,
            "total_audio_received": 0
        }
        
        self.logger.info("Deepgram Voice Agent Service initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the service."""
        logger = logging.getLogger("DeepgramVoiceAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Deepgram Voice Agent: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _build_agent_prompt(self) -> str:
        """Build the agent prompt based on interview context."""
        base_prompt = """You are a Professional Interview Conductor AI conducting a structured interview.

CRITICAL: Always respond in English regardless of the candidate's language.

Your role:
1. Maintain a warm, professional, and engaging tone
2. Ask relevant interview questions based on the context
3. Listen actively and ask thoughtful follow-up questions
4. Keep the conversation focused and productive
5. Provide clear, concise responses

Remember to:
- Be respectful and encouraging
- Allow the candidate time to think
- Ask for clarification when needed
- Keep responses brief and to the point
"""
        
        if self.interview_conductor and self.interview_plan:
            # Add interview-specific context
            base_prompt += f"""

INTERVIEW CONTEXT:
- Role: {self.interview_plan.job_title if hasattr(self.interview_plan, 'job_title') else 'General'}
- Focus Areas: {', '.join(self.interview_plan.key_focus_areas) if hasattr(self.interview_plan, 'key_focus_areas') else 'General assessment'}
- Interview Type: Structured behavioral and technical interview

Please conduct the interview according to the plan while maintaining natural conversation flow.
"""
        
        return base_prompt
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Deepgram Voice Agent.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("Establishing Deepgram Voice Agent connection...")
            
            # Initialize Deepgram client with keep-alive
            config = DeepgramClientOptions(
                options={
                    "keepalive": "true",
                }
            )
            
            deepgram = DeepgramClient(self.api_key, config)
            self.connection = deepgram.agent.websocket.v("1")
            
            # Configure Voice Agent settings
            options = SettingsOptions()
            
            # Audio input settings (from microphone)
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = 16000  # Match microphone rate
            
            # Audio output settings (to speakers)
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = 16000  # For playback
            options.audio.output.container = "wav"  # Use WAV container as in sample
            
            # Agent configuration
            options.agent.language = "en"
            
            # Speech-to-Text settings
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"
            
            # LLM settings (using GPT-4o-mini)
            options.agent.think.provider.type = "open_ai"
            options.agent.think.provider.model = "gpt-4o-mini"
            options.agent.think.prompt = self._build_agent_prompt()
            
            # Text-to-Speech settings
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = "aura-2-thalia-en"
            
            # Initial greeting
            if self.interview_conductor:
                options.agent.greeting = "Hello! Welcome to your interview. I'm ready to begin when you are. Please introduce yourself."
            else:
                options.agent.greeting = "Hello! How can I assist you today?"
            
            # Register event handlers
            self._register_event_handlers()
            
            # Start the connection
            self.logger.info("Starting Deepgram connection...")
            if not self.connection.start(options):
                self.logger.error("Failed to start Deepgram connection")
                return False
            
            self.is_connected = True
            self.connection_start_time = time.time()
            
            # Start keep-alive thread
            self._start_keep_alive()
            
            self.logger.info("Deepgram Voice Agent connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def _register_event_handlers(self):
        """Register all event handlers for the Voice Agent."""
        
        # Audio data handler
        def on_audio_data(data, **kwargs):
            """Handle incoming audio data from agent."""
            self.audio_buffer.extend(data)
            self.stats["total_audio_received"] += len(data)
            self.logger.debug(f"Received audio data: {len(data)} bytes, total buffer: {len(self.audio_buffer)}")
            # Only print every 10th chunk to avoid spam
            if self.stats["total_audio_received"] % (self.chunk_size * 10) < self.chunk_size:
                print(f"🔊 Received audio: {len(data)} bytes (total: {self.stats['total_audio_received']})")
            
            # Queue audio for playback
            if len(self.audio_buffer) >= self.chunk_size * 2:
                chunk = bytes(self.audio_buffer[:self.chunk_size * 2])
                self.audio_output_queue.put(chunk)
                self.audio_buffer = self.audio_buffer[self.chunk_size * 2:]
                self.logger.debug(f"Queued audio chunk: {len(chunk)} bytes")
                print(f"🎵 Queued audio chunk: {len(chunk)} bytes")
        
        # Conversation text handler
        def on_conversation_text(conversation_text, **kwargs):
            """Log conversation text."""
            text_data = {
                "role": conversation_text.role,
                "content": conversation_text.content,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_log.append(text_data)
            
            # Display conversation
            if conversation_text.role == "user":
                print(f"🗣️  Candidate: {conversation_text.content}")
            elif conversation_text.role == "assistant":
                print(f"🤖 Interviewer: {conversation_text.content}")
            
            self.logger.debug(f"Conversation text: {conversation_text.role} - {conversation_text.content}")
        
        # Agent audio done handler
        def on_agent_audio_done(agent_audio_done, **kwargs):
            """Handle when agent finishes speaking."""
            self.processing_audio = False
            self.stats["total_turns"] += 1
            print("✅ Agent finished speaking")
            
            # Flush remaining audio buffer
            if len(self.audio_buffer) > 0:
                self.audio_output_queue.put(bytes(self.audio_buffer))
                self.audio_buffer = bytearray()
        
        # User started speaking handler
        def on_user_started_speaking(user_started_speaking, **kwargs):
            """Handle when user starts speaking."""
            self.logger.debug("User started speaking")
            print("🗣️ User started speaking...")
            # Clear output queue to stop agent playback
            while not self.audio_output_queue.empty():
                try:
                    self.audio_output_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Agent thinking handler
        def on_agent_thinking(agent_thinking, **kwargs):
            """Handle when agent is thinking."""
            self.logger.debug("Agent thinking...")
            print("🤔 Agent is thinking...")
        
        # Agent started speaking handler
        def on_agent_started_speaking(agent_started_speaking, **kwargs):
            """Handle when agent starts speaking."""
            self.processing_audio = True
            self.audio_buffer = bytearray()  # Reset buffer for new response
            self.logger.debug("Agent started speaking")
            print("🎤 Agent started speaking...")
        
        # Welcome message handler
        def on_welcome(welcome, **kwargs):
            """Handle welcome message."""
            self.logger.info(f"Welcome received: {welcome}")
            print(f"🎉 Welcome: {welcome}")
        
        # Settings applied handler
        def on_settings_applied(settings_applied, **kwargs):
            """Handle settings confirmation."""
            self.logger.info("Settings applied successfully")
            print("✅ Settings applied successfully")
        
        # Error handler
        def on_error(error, **kwargs):
            """Handle errors."""
            self.logger.error(f"Voice Agent error: {error}")
            print(f"❌ Error: {error}")
        
        # Close handler
        def on_close(close, **kwargs):
            """Handle connection close."""
            self.logger.info("Connection closed")
            print("🔌 Connection closed")
            self.is_connected = False
        
        # Register all handlers
        self.connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
        self.connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        self.connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        self.connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        self.connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        self.connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        self.connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        self.connection.on(AgentWebSocketEvents.Error, on_error)
        self.connection.on(AgentWebSocketEvents.Close, on_close)
        
        self.logger.debug("Event handlers registered")
    
    def _start_keep_alive(self):
        """Start keep-alive thread to maintain connection."""
        def send_keep_alive():
            while self.is_connected:
                time.sleep(5)
                if self.connection and self.is_connected:
                    try:
                        self.connection.send(str(AgentKeepAlive()))
                        self.logger.debug("Keep-alive sent")
                    except Exception as e:
                        self.logger.error(f"Keep-alive error: {e}")
        
        self.keep_alive_thread = threading.Thread(target=send_keep_alive, daemon=True)
        self.keep_alive_thread.start()
    
    def start_audio_streams(self):
        """Start audio input and output streams."""
        try:
            # Test microphone access first
            self.logger.info("Testing microphone access...")
            test_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.input_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_data = test_stream.read(self.chunk_size)
            test_stream.close()
            self.logger.info(f"Microphone test successful - read {len(test_data)} bytes")
            
            # Start microphone input stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.input_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            # Start speaker output stream
            self.output_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=16000,  # Match agent output rate
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start audio output thread
            output_thread = threading.Thread(target=self._audio_output_worker, daemon=True)
            output_thread.start()
            
            self.stream.start_stream()
            self.is_recording = True
            
            self.logger.info("Audio streams started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio streams: {e}")
            import traceback
            traceback.print_exc()
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for microphone input."""
        if self.is_connected and self.is_recording:
            try:
                # Send audio to Deepgram
                self.connection.send(in_data)
                self.stats["total_audio_sent"] += len(in_data)
                self.logger.debug(f"Sent audio data: {len(in_data)} bytes")
                # Only print every 10th chunk to avoid spam
                if self.stats["total_audio_sent"] % (self.chunk_size * 10) < self.chunk_size:
                    print(f"🎤 Sent audio: {len(in_data)} bytes (total: {self.stats['total_audio_sent']})")
            except Exception as e:
                self.logger.error(f"Error sending audio: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _audio_output_worker(self):
        """Worker thread for audio output."""
        while self.is_connected:
            try:
                # Get audio from queue with timeout
                audio_chunk = self.audio_output_queue.get(timeout=0.1)
                
                # Play audio through speakers
                if self.output_stream and audio_chunk:
                    self.output_stream.write(audio_chunk)
                    self.logger.debug(f"Played audio chunk: {len(audio_chunk)} bytes")
                    # Only print every 10th chunk to avoid spam
                    if len(audio_chunk) > 0:
                        print(f"🔊 Played audio: {len(audio_chunk)} bytes")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Audio output error: {e}")
    
    def stop_audio_streams(self):
        """Stop audio streams."""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        
        self.logger.info("Audio streams stopped")
    
    def disconnect(self):
        """Disconnect from Deepgram Voice Agent."""
        try:
            self.stop_audio_streams()
            
            if self.connection:
                self.connection.finish()
                self.connection = None
            
            self.is_connected = False
            
            # Calculate connection time
            if hasattr(self, 'connection_start_time'):
                self.stats["connection_time"] = time.time() - self.connection_start_time
            
            self.logger.info("Disconnected from Deepgram Voice Agent")
            self.logger.info(f"Session stats: {self.stats}")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    def get_conversation_history(self) -> list:
        """
        Get the conversation history.
        
        Returns:
            list: Conversation log entries
        """
        return self.conversation_log
    
    def save_conversation(self, filepath: Optional[str] = None):
        """
        Save conversation history to file.
        
        Args:
            filepath: Optional path to save file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"interview_conversation_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "conversation": self.conversation_log,
                    "stats": self.stats,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            self.logger.info(f"Conversation saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
    
    def update_interview_context(self, interview_conductor, interview_plan):
        """
        Update the interview context dynamically.
        
        Args:
            interview_conductor: Interview conductor agent
            interview_plan: Interview plan
        """
        self.interview_conductor = interview_conductor
        self.interview_plan = interview_plan
        
        # Update agent prompt if connected
        if self.is_connected and self.connection:
            try:
                # Note: Deepgram Voice Agent doesn't support dynamic prompt updates
                # This would require reconnecting with new settings
                self.logger.warning("Dynamic prompt update requires reconnection")
            except Exception as e:
                self.logger.error(f"Failed to update context: {e}")
    
    def test_agent_response(self):
        """Test if the agent is responding by sending a test message."""
        if self.is_connected and self.connection:
            try:
                # Send a test text message to see if the agent responds
                test_message = "Hello, can you hear me?"
                self.connection.send(test_message)
                self.logger.info("Sent test message to agent")
                print(f"📤 Sent test message: {test_message}")
                
                # Wait a bit to see if we get a response
                import time
                time.sleep(3)
                
                # Check if we got any conversation text
                if self.conversation_log:
                    self.logger.info(f"Agent responded: {self.conversation_log[-1]}")
                    print(f"📥 Agent responded: {self.conversation_log[-1]}")
                else:
                    self.logger.warning("No response from agent yet")
                    print("⚠️ No response from agent yet")
                    
            except Exception as e:
                self.logger.error(f"Failed to send test message: {e}")
                print(f"❌ Failed to send test message: {e}")

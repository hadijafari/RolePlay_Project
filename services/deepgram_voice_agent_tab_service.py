"""
Deepgram Voice Agent Service with Tab-Key Recording
Records audio while TAB is held, then sends to Deepgram Voice Agent
"""

import os
import sys
import time
import logging
import threading
import asyncio
import queue
import json
import wave
import io
from pathlib import Path
from typing import Optional, Dict, Any, List
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
    import keyboard
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Deepgram Voice Agent Service: Missing required dependency: {e}")
    print("Please install dependencies: pip install deepgram-sdk pyaudio keyboard python-dotenv")
    sys.exit(1)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class DeepgramVoiceAgentTabService:
    """
    Service for managing voice conversations using Deepgram Voice Agent with TAB key recording.
    
    This service provides:
    - TAB key controlled recording
    - Send complete audio to Deepgram Voice Agent
    - Receive and play response
    """
    
    def __init__(self, interview_conductor=None, interview_plan=None):
        """
        Initialize Deepgram Voice Agent service with TAB recording.
        
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
        self.keep_alive_thread = None
        
        # Audio configuration
        self.audio = pyaudio.PyAudio()
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk_size = 1024
        
        # Recording state
        self.is_recording = False
        self.recording_frames = []
        self.recording_stream = None
        self.recording_lock = threading.Lock()
        
        # Audio output handling
        self.audio_output_queue = queue.Queue()
        self.audio_buffer = bytearray()
        self.output_stream = None
        self.output_thread = None
        
        # Conversation tracking
        self.conversation_log = []
        self.current_turn_count = 0
        self.waiting_for_response = False
        
        # Statistics
        self.stats = {
            "total_turns": 0,
            "connection_time": 0,
            "total_recordings": 0
        }
        
        # Progressive streaming (send while holding TAB)
        self.progressive_streaming_enabled = True
        self.segment_buffer = bytearray()
        self.segment_lock = threading.Lock()
        self.segment_sender_thread = None
        self.segment_running = False
        self.segment_seconds = 6  # target duration per segment
        self.bytes_per_second = int(self.rate * 2 * self.channels)  # 16-bit mono
        self.segment_bytes_target = self.bytes_per_second * self.segment_seconds
        
        # Nudge mechanism for stalled responses
        self.last_send_time = None
        self.nudge_sent = False
        self.agent_ready = False
        
        self.logger.info("Deepgram Voice Agent TAB Service initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the service."""
        logger = logging.getLogger("DeepgramVoiceAgentTab")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Deepgram TAB Agent: %(levelname)s - %(message)s'
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
            
            # Audio input settings (from recorded audio)
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = 24000
            
            # Audio output settings (to speakers)
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = 24000
            options.audio.output.container = "wav"
            
            # Agent configuration
            options.agent.language = "en"
            
            # Speech-to-Text settings
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"
            
            # LLM settings - OpenAI GPT-4o-mini
            options.agent.think.provider.type = "open_ai"
            options.agent.think.provider.model = "gpt-4o-mini"
            options.agent.think.prompt = self._build_agent_prompt()
            
            # Text-to-Speech settings
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = "aura-2-thalia-en"
            
            # Initial greeting
            if self.interview_conductor:
                options.agent.greeting = "Hello! Welcome to your interview. I'm ready to begin when you are. Please hold TAB and introduce yourself."
            else:
                options.agent.greeting = "Hello! How can I assist you today? Hold TAB to speak."
            
            # Register event handlers
            self._register_event_handlers()
            
            # Debug: Log the configuration being sent
            self.logger.info(f"Agent configuration: {options.agent.__dict__}")
            self.logger.info(f"Think provider: {options.agent.think.provider}")
            
            # Start the connection
            self.logger.info("Starting Deepgram connection...")
            if not self.connection.start(options):
                self.logger.error("Failed to start Deepgram connection")
                return False
            
            self.is_connected = True
            self.connection_start_time = time.time()
            
            # Start keep-alive thread
            self._start_keep_alive()
            
            # Start audio output thread
            self.output_thread = threading.Thread(target=self._audio_output_worker, daemon=True)
            self.output_thread.start()
            
            # Initialize output stream
            self.output_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=24000,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info("Deepgram Voice Agent connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _register_event_handlers(self):
        """Register all event handlers for the Voice Agent."""
        
        # Audio data handler - receives audio from agent
        def on_audio_data(_, data, **kwargs):
            """Handle incoming audio data from agent."""
            self.audio_buffer.extend(data)
            
            # Queue audio for playback
            if len(self.audio_buffer) >= self.chunk_size * 2:
                chunk = bytes(self.audio_buffer[:self.chunk_size * 2])
                self.audio_output_queue.put(chunk)
                self.audio_buffer = self.audio_buffer[self.chunk_size * 2:]
        
        # Conversation text handler
        def on_conversation_text(_, conversation_text, **kwargs):
            """Log conversation text."""
            text_data = {
                "role": conversation_text.role,
                "content": conversation_text.content,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_log.append(text_data)
            
            # Display conversation
            if conversation_text.role == "user":
                print(f"🗣️  You: {conversation_text.content}")
            elif conversation_text.role == "assistant":
                print(f"🤖 Interviewer: {conversation_text.content}")
                # Consider response arrived as soon as assistant text is available
                if self.waiting_for_response:
                    self.waiting_for_response = False
                    self.logger.debug("Turn completed by ConversationText (assistant)")
        
        # Agent audio done handler
        def on_agent_audio_done(_, agent_audio_done, **kwargs):
            """Handle when agent finishes speaking."""
            # Flush remaining audio buffer
            if len(self.audio_buffer) > 0:
                self.audio_output_queue.put(bytes(self.audio_buffer))
                self.audio_buffer = bytearray()
            
            self.waiting_for_response = False
            self.stats["total_turns"] += 1
            print("✅ Agent finished speaking - Hold TAB to respond")
        
        # User started speaking handler
        def on_user_started_speaking(_, user_started_speaking, **kwargs):
            """Handle when user starts speaking."""
            # Clear output queue to stop agent playback
            while not self.audio_output_queue.empty():
                try:
                    self.audio_output_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Agent thinking handler
        def on_agent_thinking(_, agent_thinking, **kwargs):
            """Handle when agent is thinking."""
            print("🤔 Agent is thinking...")
            self.logger.info(f"Agent thinking event: {agent_thinking}")
            # Reset nudge flag when agent starts thinking
            self.nudge_sent = False
        
        # Agent started speaking handler
        def on_agent_started_speaking(_, agent_started_speaking, **kwargs):
            """Handle when agent starts speaking."""
            self.audio_buffer = bytearray()  # Reset buffer for new response
            print("🎤 Agent is speaking...")
            self.logger.info(f"Agent started speaking event: {agent_started_speaking}")
            # Release wait as soon as agent begins speaking
            if self.waiting_for_response:
                self.waiting_for_response = False
                self.logger.debug("Turn completed by AgentStartedSpeaking")
        
        # History handler (prevents duplicate Unknown/Unhandled logs)
        def on_history(_, history, **kwargs):
            """Handle Deepgram History events without duplicate prints."""
            try:
                # History typically includes role/content; we store it but avoid printing
                entry = {
                    "role": getattr(history, "role", None),
                    "content": getattr(history, "content", None),
                    "timestamp": datetime.now().isoformat(),
                    "source": "history"
                }
                self.conversation_log.append(entry)
                self.logger.debug("History event captured")
            except Exception:
                # Best-effort: do not raise; just avoid noisy output
                pass

        # Welcome message handler
        def on_welcome(_, welcome, **kwargs):
            """Handle welcome message."""
            self.logger.info(f"Welcome received: {welcome}")
            print(f"🎉 Connected to Deepgram Voice Agent")
        
        # Settings applied handler
        def on_settings_applied(_, settings_applied, **kwargs):
            """Handle settings confirmation."""
            self.logger.info("Settings applied successfully")
            self.agent_ready = True
            print("🟢 Agent is ready for conversation")
        
        # Error handler
        def on_error(_, error, **kwargs):
            """Handle errors."""
            self.logger.error(f"Voice Agent error: {error}")
            print(f"❌ Error: {error}")
            # Prevent indefinite waiting on error
            if self.waiting_for_response:
                self.waiting_for_response = False
                self.logger.debug("Turn aborted due to Error event")
        
        # Warning handler
        def on_warning(_, warning, **kwargs):
            """Handle warnings."""
            self.logger.warning(f"Voice Agent warning: {warning}")
            print(f"⚠️ Warning: {warning}")
        
        # Close handler
        def on_close(_, close, **kwargs):
            """Handle connection close."""
            self.logger.info("Connection closed")
            self.is_connected = False
        
        # Unhandled event handler - catches any events we're not explicitly handling
        def on_unhandled(_, unhandled, **kwargs):
            """Handle unhandled events."""
            try:
                raw = getattr(unhandled, "raw", "")
                # Suppress duplicate logs for History events
                if isinstance(raw, str) and ('"type":"History"' in raw or "'type': 'History'" in raw):
                    return
            except Exception:
                pass
            self.logger.info(f"Unhandled event: {unhandled}")
            print(f"📝 Unhandled event: {unhandled}")
        
        # Register all handlers
        self.connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
        self.connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        self.connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        self.connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        self.connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        # History is surfaced by the server to replay context
        try:
            self.connection.on(AgentWebSocketEvents.History, on_history)
        except Exception:
            # Older SDKs may not expose History; ignore silently
            pass
        self.connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        self.connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        self.connection.on(AgentWebSocketEvents.Error, on_error)
        # self.connection.on(AgentWebSocketEvents.Warning, on_warning)  # Uncomment if Warning event exists
        self.connection.on(AgentWebSocketEvents.Close, on_close)
        self.connection.on(AgentWebSocketEvents.Unhandled, on_unhandled)
        
        self.logger.debug("Event handlers registered")
    
    def _start_keep_alive(self):
        """Start keep-alive thread to maintain connection."""
        def send_keep_alive():
            while self.is_connected:
                time.sleep(3)  # Reduced from 5 to 3 seconds
                if self.connection and self.is_connected:
                    try:
                        self.connection.send(str(AgentKeepAlive()))
                        self.logger.debug("Keep-alive sent")
                        # Check if we need to send a nudge
                        self._check_and_send_nudge()
                    except Exception as e:
                        self.logger.error(f"Keep-alive error: {e}")
        
        self.keep_alive_thread = threading.Thread(target=send_keep_alive, daemon=True)
        self.keep_alive_thread.start()
    
    def _check_and_send_nudge(self):
        """Send a nudge if no AgentThinking received within 2 seconds of last send."""
        if (self.waiting_for_response and 
            self.last_send_time and 
            not self.nudge_sent and
            time.time() - self.last_send_time > 2.0):
            
            try:
                # Send a 200ms silence nudge
                silence_ms = 200
                bytes_per_sample = 2  # 16-bit linear16
                silence_bytes = int(self.rate * (silence_ms / 1000.0)) * bytes_per_sample
                self.connection.send(b"\x00" * silence_bytes)
                self.nudge_sent = True
                self.logger.info("Sent nudge to wake up agent")
                print("📢 Sent nudge to agent")
            except Exception as e:
                self.logger.error(f"Error sending nudge: {e}")
    
    def start_recording(self):
        """Start recording audio when TAB is pressed."""
        with self.recording_lock:
            if self.is_recording or self.waiting_for_response:
                return
            
            try:
                self.is_recording = True
                self.recording_frames = []
                
                # Open recording stream
                self.recording_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # Start recording thread
                recording_thread = threading.Thread(target=self._record_audio, daemon=True)
                recording_thread.start()
                
                # Start progressive sender if enabled
                if self.progressive_streaming_enabled:
                    self.segment_running = True
                    self.segment_sender_thread = threading.Thread(target=self._segment_sender_worker, daemon=True)
                    self.segment_sender_thread.start()
                
                print("🎤 Recording... (release TAB to send)")
                
            except Exception as e:
                self.logger.error(f"Failed to start recording: {e}")
                self.is_recording = False
    
    def _record_audio(self):
        """Record audio in a separate thread."""
        while self.is_recording:
            try:
                data = self.recording_stream.read(self.chunk_size, exception_on_overflow=False)
                self.recording_frames.append(data)
                # Feed progressive segment buffer when enabled
                if self.progressive_streaming_enabled:
                    with self.segment_lock:
                        self.segment_buffer.extend(data)
            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break
    
    def _segment_sender_worker(self):
        """Continuously flush ~6s segments to Deepgram while TAB is held."""
        while self.segment_running and self.is_connected and self.connection:
            try:
                to_send = b""
                with self.segment_lock:
                    if len(self.segment_buffer) >= self.segment_bytes_target:
                        to_send = bytes(self.segment_buffer[:self.segment_bytes_target])
                        del self.segment_buffer[:self.segment_bytes_target]
                if to_send:
                    # Stream this segment in chunks
                    chunk_size = 8192
                    total_sent = 0
                    for i in range(0, len(to_send), chunk_size):
                        chunk = to_send[i:i + chunk_size]
                        try:
                            self.connection.send(chunk)
                            total_sent += len(chunk)
                            self.logger.debug(f"Sent progressive chunk: {len(chunk)} bytes")
                            time.sleep(0.01)
                        except Exception as send_error:
                            self.logger.error(f"Error sending progressive chunk: {send_error}")
                            break
                    if total_sent:
                        self.logger.info(f"Progressive segment sent: {total_sent} bytes")
                else:
                    time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"Segment sender error: {e}")
                time.sleep(0.1)

    def _flush_final_segment_and_wait(self):
        """Flush remaining progressive audio and wait for response."""
        if not self.is_connected or not self.connection:
            print("❌ Not connected to Deepgram")
            return
        try:
            self.waiting_for_response = True
            print("📤 Sending audio to Deepgram...")
            self.logger.info(f"Connection status: connected={self.is_connected}")
            
            # Collect remaining unflushed audio
            with self.segment_lock:
                audio_data = bytes(self.segment_buffer)
                self.segment_buffer.clear()
            
            # Log audio details
            self.logger.info(f"Sending audio: {len(audio_data)} bytes, sample rate: {self.rate}, channels: {self.channels}")
            
            if not self.is_connected:
                print("❌ Connection lost before sending audio")
                self.waiting_for_response = False
                return
            
            # Send remaining audio in chunks
            chunk_size = 8192
            total_sent = 0
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                try:
                    self.connection.send(chunk)
                    total_sent += len(chunk)
                    self.logger.debug(f"Sent final chunk: {len(chunk)} bytes")
                    time.sleep(0.01)
                except Exception as send_error:
                    self.logger.error(f"Error sending final chunk: {send_error}")
                    print(f"❌ Error sending final audio chunk: {send_error}")
                    break
            print(f"✅ Sent {total_sent} bytes to Deepgram")
            
            # Append brief silence to help VAD finalize
            try:
                silence_ms = 400
                bytes_per_sample = 2  # 16-bit linear16
                silence_bytes = int(self.rate * (silence_ms / 1000.0)) * bytes_per_sample
                self.connection.send(b"\x00" * silence_bytes)
                self.logger.debug("Sent 400ms silence tail")
            except Exception as e:
                self.logger.error(f"Error sending silence tail: {e}")
            
            print("⏳ Waiting for response...")
            
            # Record when we finished sending for nudge mechanism
            self.last_send_time = time.time()
            self.nudge_sent = False
            
            timeout_start = time.time()
            timeout = 30
            while self.waiting_for_response and time.time() - timeout_start < timeout:
                time.sleep(0.1)
            if self.waiting_for_response:
                print("⚠️ Response timeout - no response received from Deepgram")
                self.waiting_for_response = False
                self.logger.warning("Response timeout after 30 seconds")
        except Exception as e:
            self.logger.error(f"Failed to flush final segment: {e}")
            print(f"❌ Failed to flush final segment: {e}")
            self.waiting_for_response = False
    
    def stop_recording(self):
        """Stop recording and send audio to Deepgram."""
        with self.recording_lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
            
            # Close recording stream
            if self.recording_stream:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
                self.recording_stream = None
            
            # Process recorded audio
            if self.recording_frames:
                print(f"⏹️ Recording stopped - {len(self.recording_frames)} chunks recorded")
                self.stats["total_recordings"] += 1
                
                # Small delay to prevent rapid TAB presses
                time.sleep(0.1)
                
                if self.progressive_streaming_enabled:
                    # Stop progressive sender and flush remaining bytes with finalization
                    self.segment_running = False
                    if self.segment_sender_thread and self.segment_sender_thread.is_alive():
                        self.segment_sender_thread.join(timeout=0.5)
                    threading.Thread(target=self._flush_final_segment_and_wait, daemon=True).start()
                else:
                    # Fallback to single-shot send
                    threading.Thread(target=self._send_audio_sync, daemon=True).start()
    
    def _send_audio_sync(self):
        """Synchronous version of sending audio to Deepgram."""
        if not self.is_connected or not self.connection:
            print("❌ Not connected to Deepgram")
            return
        
        try:
            self.waiting_for_response = True
            print("📤 Sending audio to Deepgram...")
            self.logger.info(f"Connection status: connected={self.is_connected}")
            
            # Convert frames to bytes
            audio_data = b''.join(self.recording_frames)
            self.recording_frames = []  # Clear frames after collecting
            
            # Log audio details
            self.logger.info(f"Sending audio: {len(audio_data)} bytes, sample rate: {self.rate}, channels: {self.channels}")
            
            # Check if connection is still alive
            if not self.is_connected:
                print("❌ Connection lost before sending audio")
                self.waiting_for_response = False
                return
            
            # Stream audio in chunks with slight pacing
            chunk_size = 8192
            total_sent = 0
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                try:
                    self.connection.send(chunk)
                    total_sent += len(chunk)
                    self.logger.debug(f"Sent chunk: {len(chunk)} bytes")
                    time.sleep(0.01)
                except Exception as send_error:
                    self.logger.error(f"Error sending chunk: {send_error}")
                    print(f"❌ Error sending audio chunk: {send_error}")
                    break

            print(f"✅ Sent {total_sent} bytes to Deepgram")

            # Append brief silence to help VAD finalize
            try:
                silence_ms = 200
                bytes_per_sample = 2  # 16-bit linear16
                silence_bytes = int(self.rate * (silence_ms / 1000.0)) * bytes_per_sample
                self.connection.send(b"\x00" * silence_bytes)
                self.logger.debug("Sent 200ms silence tail")
            except Exception as e:
                self.logger.error(f"Error sending silence tail: {e}")
            
            print("⏳ Waiting for response...")
            
            # Record when we finished sending for nudge mechanism
            self.last_send_time = time.time()
            self.nudge_sent = False
            
            # Add a timeout mechanism (wait max 30 seconds for response)
            timeout_start = time.time()
            timeout = 30  # seconds
            
            while self.waiting_for_response and time.time() - timeout_start < timeout:
                time.sleep(0.1)
            
            if self.waiting_for_response:
                print("⚠️ Response timeout - no response received from Deepgram")
                self.waiting_for_response = False
                self.logger.warning("Response timeout after 30 seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}")
            print(f"❌ Failed to send audio: {e}")
            self.waiting_for_response = False
            import traceback
            traceback.print_exc()
    
    async def _send_audio_to_deepgram(self):
        """Send recorded audio to Deepgram Voice Agent (async version)."""
        if not self.is_connected or not self.connection:
            print("❌ Not connected to Deepgram")
            return
        
        try:
            self.waiting_for_response = True
            print("📤 Sending audio to Deepgram...")
            
            # Convert frames to bytes
            audio_data = b''.join(self.recording_frames)
            
            # Send audio in chunks (similar to sample code)
            chunk_size = 8192
            total_sent = 0
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.connection.send(chunk)
                total_sent += len(chunk)
            
            print(f"✅ Sent {total_sent} bytes to Deepgram")
            print("⏳ Waiting for response...")
            
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}")
            print(f"❌ Failed to send audio: {e}")
            self.waiting_for_response = False
    
    def _audio_output_worker(self):
        """Worker thread for audio output."""
        while self.is_connected:
            try:
                # Get audio from queue with timeout
                audio_chunk = self.audio_output_queue.get(timeout=0.1)
                
                # Play audio through speakers
                if self.output_stream and audio_chunk:
                    self.output_stream.write(audio_chunk)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Audio output error: {e}")
    
    def setup_keyboard_handlers(self):
        """Set up TAB key handlers for recording."""
        keyboard.on_press_key('tab', lambda e: self.start_recording())
        keyboard.on_release_key('tab', lambda e: self.stop_recording())
        print("🎮 Keyboard handlers set - Hold TAB to record")
    
    def disconnect(self):
        """Disconnect from Deepgram Voice Agent."""
        try:
            self.is_connected = False
            
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Close streams
            if self.recording_stream:
                self.recording_stream.close()
            
            if self.output_stream:
                self.output_stream.close()
            
            # Disconnect from Deepgram
            if self.connection:
                self.connection.finish()
                self.connection = None
            
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
        self.logger.info("Interview context updated")

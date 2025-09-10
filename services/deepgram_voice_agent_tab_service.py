"""
Simplified Deepgram Voice Agent Service with Tab-Key Recording
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
    Simplified service for managing voice conversations using Deepgram Voice Agent with TAB key recording.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # API Configuration
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in .env file")
        
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
        self.waiting_for_response = False
        
        # Statistics
        self.stats = {
            "total_turns": 0,
            "connection_time": 0,
            "total_recordings": 0
        }
        
        self.logger.info("Simplified Deepgram Voice Agent TAB Service initialized")
    
    def _get_hardcoded_system_prompt(self) -> str:
        """Return the complete hardcoded system prompt with all 25 questions."""
        return """You are conducting a natural Electronic Engineering interview conversation. You need to ask these questions in this exact order, but make it sound like a normal conversation:

- Tell me about your experience with microcontroller programming and which platforms you've worked with.
- How do you approach debugging embedded C code when dealing with hardware-specific issues?
- Explain the difference between polling and interrupt-driven programming in microcontrollers.
- What factors do you consider when selecting a microcontroller for a new project?
- How do you handle memory management in resource-constrained embedded systems?
- Describe your experience with communication protocols like SPI, I2C, and UART.
- What tools do you use for PCB design and why do you prefer them?
- How do you ensure signal integrity in high-speed digital PCB designs?
- Explain the importance of ground planes and power distribution in PCB layout.
- What considerations do you make for EMI/EMC compliance in PCB design?
- How do you approach component placement and routing optimization on a PCB?
- Describe your experience with multi-layer PCB design and stackup considerations.
- What is your process for PCB design rule checking and design verification?
- How do you handle thermal management in PCB designs with high-power components?
- Explain the differences between through-hole and surface-mount components and when to use each.
- What experience do you have with analog circuit design and mixed-signal PCBs?
- How do you approach power supply design for embedded systems?
- Describe your experience with real-time operating systems (RTOS) in embedded applications.
- What debugging tools and techniques do you use for embedded systems development?
- How do you ensure code reliability and fault tolerance in critical embedded applications?
- Explain your approach to low-power design in battery-operated devices.
- What experience do you have with wireless communication protocols in embedded systems?
- How do you handle version control and documentation for hardware and firmware projects?
- Describe a challenging embedded systems project you've worked on and how you solved the technical issues.
- What trends do you see in embedded systems and PCB design, and how do you stay current with technology?

CRITICAL INSTRUCTIONS:
- Ask these questions in the exact order listed above
- Make it sound like a natural conversation - don't say question numbers or "first question", "second question", etc.
- After each answer, acknowledge their response naturally and move to the next topic
- Do NOT ask follow-up questions or elaborations
- Do NOT skip questions or ask them out of order
- Do NOT create your own questions
- Keep the conversation flowing naturally
- Start with: "Tell me about your experience with microcontroller programming and which platforms you've worked with."

You are a friendly, professional interviewer conducting an Electronic Engineering interview focused on Microcontroller Programming, PCB Design, and Embedded Systems."""
    
    async def start_interview(self):
        """Start the interview with keyboard controls."""
        import keyboard
        
        print("🎤 Interview started! Hold TAB to speak, release to send")
        print("❌ Press Ctrl+C to exit")
        
        try:
            # Set up keyboard event handlers
            keyboard.on_press_key('tab', lambda _: self.start_recording())
            keyboard.on_release_key('tab', lambda _: self.stop_recording())
            
            # Keep the program running
            while self.is_connected:
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            print("\n🛑 Interview stopped by user")
        except Exception as e:
            print(f"❌ Error in interview loop: {e}")
            self.logger.error(f"Interview loop error: {e}")
        finally:
            # Clean up keyboard hooks
            try:
                keyboard.unhook_all()
            except:
                pass
    
    async def disconnect(self):
        """Disconnect from Deepgram and clean up resources."""
        print("🔌 Disconnecting from Deepgram...")
        
        self.is_connected = False
        
        # Close connection
        if self.connection:
            try:
                self.connection.finish()
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
        
        # Clean up audio resources
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception as e:
                self.logger.error(f"Error closing output stream: {e}")
        
        if self.recording_stream:
            try:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
            except Exception as e:
                self.logger.error(f"Error closing recording stream: {e}")
        
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating audio: {e}")
        
        print("✅ Disconnected successfully")
    
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
            
            # LLM settings - OpenAI GPT-4o-mini with hardcoded prompt
            options.agent.think.provider.type = "open_ai"
            options.agent.think.provider.model = "gpt-4o-mini"
            options.agent.think.prompt = self._get_hardcoded_system_prompt()
            
            # Text-to-Speech settings
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = "aura-2-thalia-en"
            
            # Initial greeting
            options.agent.greeting = (
                "Hello! Welcome to your interview. "
                "I'm going to ask you 25 questions about microcontroller programming and PCB design. "
                "Please hold TAB to speak and release when you're done answering each question. "
                "Let's begin with the first question."
            )
            
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
            print("✅ DEEPGRAM CONNECTION ESTABLISHED")
            
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
                if self.waiting_for_response:
                    self.waiting_for_response = False
                    self.logger.debug("Turn completed by ConversationText (assistant)")
        
        # Agent audio done handler
        def on_agent_audio_done(_, agent_audio_done, **kwargs):
            """Handle when agent finishes speaking."""
            print(f"\n🎯 AGENT_AUDIO_DONE EVENT TRIGGERED!")
            
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
            print("🎤 USER STARTED SPEAKING - Agent must have finished")
            
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
        
        # Welcome message handler
        def on_welcome(_, welcome, **kwargs):
            """Handle welcome message."""
            self.logger.info(f"Welcome received: {welcome}")
            print(f"🎉 Connected to Deepgram Voice Agent")
        
        # Settings applied handler
        def on_settings_applied(_, settings_applied, **kwargs):
            """Handle settings confirmation."""
            self.logger.info("Settings applied successfully")
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
        print("✅ Event handlers registered")
        
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
                    except Exception as e:
                        self.logger.error(f"Keep-alive error: {e}")
        
        self.keep_alive_thread = threading.Thread(target=send_keep_alive, daemon=True)
        self.keep_alive_thread.start()
    
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
            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break
    
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
                
                # Send audio to Deepgram
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
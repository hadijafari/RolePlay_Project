"""
Enhanced AI Interview Platform - Audio Recording System
MODIFIED VERSION with Advanced Document Analysis and Interview Planning

New Features:
- Document intelligence for resume and job description analysis  
- AI-powered interview plan generation
- Context-aware interview conductor agent
- Structured context injection (fixes critical context passing issue)
- Multi-modal document processing
- Dynamic interview management

Spacebar-controlled audio recording with comprehensive interview management.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

try:
    import pyaudio
    import keyboard
    import asyncio
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies: pip install pyaudio keyboard")
    sys.exit(1)

# Import original services
from config.settings import AudioConfig, STTConfig
from services.stt_service import STTService
from services.tts_service import TTSService

# Import enhanced services and agents
try:
    from agents.document_intelligence_agent import DocumentIntelligenceAgent
    from agents.interview_conductor_agent import InterviewConductorAgent
    from services.interview_planning_service import InterviewPlanningService
    from services.context_injection_service import ContextInjectionService
    from models.interview_models import (
        ResumeAnalysis, JobDescriptionAnalysis, CandidateMatch, 
        InterviewPlan, InterviewState, ProcessingResult
    )
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced interview features loaded successfully")
except ImportError as e:
    print(f"⚠️  Enhanced features not available: {e}")
    print("⚠️  Running in basic mode without document analysis")
    ENHANCED_FEATURES_AVAILABLE = False
    # Set fallback classes
    DocumentIntelligenceAgent = None
    InterviewConductorAgent = None
    InterviewPlanningService = None


class EnhancedInterviewPlatform:
    """
    Enhanced AI Interview Platform with document analysis and interview planning.
    
    This platform extends the original audio recording functionality with:
    - Document intelligence for resume and job description analysis
    - AI-powered interview planning and question generation
    - Context-aware interview conductor that maintains conversation flow
    - Real-time interview progress tracking and evaluation
    
    Key improvements:
    - Fixed context passing issue between agents
    - Structured data handling with Pydantic models
    - Multi-modal document processing capabilities
    - Dynamic interview management with real-time adaptation
    """
    
    def __init__(self, minimal_logging: bool = True):
        self.minimal_logging = minimal_logging
        
        # Document analysis state
        self.current_resume_path = None
        self.current_job_description_path = None
        self.resume_analysis = None
        self.job_description_analysis = None
        self.candidate_match = None
        self.interview_plan = None
        
        # Interview state  
        self.interview_active = False
        self.interview_conductor = None
        self.interview_state = None
        
        # Initialize original services
        self._initialize_core_services()
        
        # Initialize enhanced services if available
        if ENHANCED_FEATURES_AVAILABLE:
            self._initialize_enhanced_services()
        
        # Initialize recorder with enhanced capabilities
        self.recorder = EnhancedAudioRecorder(
            stt_service=self.stt_service,
            tts_service=self.tts_service,
            document_agent=getattr(self, 'document_agent', None),
            interview_conductor=self.interview_conductor,
            interview_plan=self.interview_plan
        )
        
        # Application state
        self.running = False
    
    def _initialize_core_services(self):
        """Initialize core STT and TTS services."""
        
        # Initialize STT service
        try:
            self.stt_service = STTService()
            if not self.minimal_logging:
                print("✅ STT service initialized")
        except Exception as e:
            print(f"⚠️  STT service initialization failed: {e}")
            self.stt_service = None
        
        # Initialize TTS service
        try:
            self.tts_service = TTSService()
            if not self.minimal_logging:
                print("✅ TTS service initialized")
        except Exception as e:
            print(f"⚠️  TTS service initialization failed: {e}")
            self.tts_service = None
    
    def _initialize_enhanced_services(self):
        """Initialize enhanced AI services for document analysis and interview planning."""
        
        try:
            # Initialize document intelligence agent
            self.document_agent = DocumentIntelligenceAgent()
            if not self.minimal_logging:
                print("✅ Document Intelligence Agent initialized")
        except Exception as e:
            print(f"⚠️  Document Intelligence Agent initialization failed: {e}")
            self.document_agent = None
        
        try:
            # Initialize interview planning service
            self.interview_planning_service = InterviewPlanningService()
            if not self.minimal_logging:
                print("✅ Interview Planning Service initialized")
        except Exception as e:
            print(f"⚠️  Interview Planning Service initialization failed: {e}")
            self.interview_planning_service = None
        
        try:
            # Initialize context injection service
            self.context_service = ContextInjectionService()
            if not self.minimal_logging:
                print("✅ Context Injection Service initialized")
        except Exception as e:
            print(f"⚠️  Context Injection Service initialization failed: {e}")
            self.context_service = None
        
        try:
            # Initialize interview conductor agent
            self.interview_conductor = InterviewConductorAgent(
                context_service=self.context_service
            )
            if not self.minimal_logging:
                print("✅ Interview Conductor Agent initialized")
        except Exception as e:
            print(f"⚠️  Interview Conductor Agent initialization failed: {e}")
            self.interview_conductor = None
    
    def display_welcome(self):
        """Display enhanced welcome message with feature overview."""
        
        print("\n" + "=" * 80)
        print("🎤 ENHANCED AI INTERVIEW PLATFORM")
        print("=" * 80)
        
        if ENHANCED_FEATURES_AVAILABLE:
            print("🧠 AI-Powered Document Intelligence & Interview Planning")
            print("🎯 Context-Aware Interview Conductor")
            print("📊 Real-Time Progress Tracking & Evaluation")
            print("🔧 Fixed Context Passing & Structured Data Handling")
            print()
            print("✨ AUTOMATIC SETUP:")
            print("   • Resume & Job Description Analysis")
            print("   • Strategic Interview Plan Generation")
            print("   • Context-Aware Question Selection")
            print("   • Dynamic Interview Management")
            print()
            print("🎮 CONTROLS:")
            print("   • SPACEBAR (hold): Record your response")
            print("   • ESC: Exit interview")
            print()
        else:
            print("📝 NOTE: Running in basic mode - upload enhanced modules for full features")
        
        print("-" * 80)
    
    # Command input handling removed - now using automatic setup
    
    def _show_help(self):
        """Show enhanced help information."""
        
        print("\n📋 ENHANCED AI INTERVIEW PLATFORM - HELP")
        print("=" * 60)
        print()
        print("🎤 AUDIO CONTROLS:")
        print("  SPACEBAR (hold)  - Record audio response")
        print("  SPACEBAR (release) - Stop recording and process")
        print("  ESC              - Exit application")
    
    def setup_keyboard_handlers(self):
        """Set up keyboard event handlers."""
        
        # Spacebar handlers for audio recording
        keyboard.on_press_key('space', self._on_spacebar_press)
        keyboard.on_release_key('space', self._on_spacebar_release)
        
        # Escape key handler
        keyboard.on_press_key('esc', self._on_escape_press)
    
    def _on_spacebar_press(self, event):
        """Handle spacebar press for recording."""
        if not self.recorder.is_recording:
            self.recorder.start_recording()
    
    def _on_spacebar_release(self, event):
        """Handle spacebar release to stop recording."""
        if self.recorder.is_recording:
            self.recorder.stop_recording()
    
    def _on_escape_press(self, event):
        """Handle escape key press."""
        print("\nInterview Platform: Shutting down...")
        self.running = False
    
    async def run(self):
        """Main application loop with enhanced features."""
        
        try:
            # Check microphone access
            if not self.recorder.check_microphone_access():
                return
            
            # Test services
            await self._test_services()
            
            # Display welcome
            self.display_welcome()
            
            # Set up keyboard handlers
            self.setup_keyboard_handlers()
            
            # Start main loop
            self.running = True
            
            if ENHANCED_FEATURES_AVAILABLE:
                # Run enhanced mode with automatic setup
                await self._run_enhanced_mode()
            else:
                # Run basic mode
                await self._run_basic_mode()
        
        except KeyboardInterrupt:
            print("\nInterview Platform: Interrupted by user")
        except Exception as e:
            print(f"Interview Platform: Unexpected error - {e}")
        finally:
            await self._cleanup()
    
    async def _test_services(self):
        """Test all available services."""
        
        print("🔧 Testing services...")
        
        # Test STT
        if self.stt_service and self.stt_service.test_connection():
            print("✅ STT service ready")
        elif self.stt_service:
            print("⚠️  STT service connection issues")
        else:
            print("❌ STT service not available")
        
        # Test TTS
        if self.tts_service and self.tts_service.test_connection():
            print("✅ TTS service ready")
        elif self.tts_service:
            print("⚠️  TTS service connection issues")
        else:
            print("❌ TTS service not available")
        
        # Test enhanced services
        if ENHANCED_FEATURES_AVAILABLE:
            if hasattr(self, 'document_agent') and self.document_agent:
                print("✅ Document Intelligence Agent ready")
            if hasattr(self, 'interview_planning_service') and self.interview_planning_service:
                print("✅ Interview Planning Service ready")
    
    async def _run_enhanced_mode(self):
        """Run application in enhanced mode with automatic setup."""
        
        # Automatically set up the interview
        setup_success = await self._automatic_interview_setup()
        
        if not setup_success:
            print("❌ Failed to set up interview automatically. Exiting...")
            self.running = False
            return
        
        # Main loop for keyboard events only (no command input)
        print("\n🎤 Interview ready! Hold SPACEBAR to record, press ESC to exit")
        try:
            while self.running:
                await asyncio.sleep(0.1)
        except:
            pass
    
    async def _run_basic_mode(self):
        """Run application in basic mode (original functionality)."""
        
        print("\n📝 Basic Mode: Audio recording and transcription only")
        print("Hold SPACEBAR to record, release to stop and transcribe")
        print("Press ESC to exit")
        
        while self.running:
            await asyncio.sleep(0.1)
    
    async def _cleanup(self):
        """Clean up all resources."""
        
        print("🧹 Cleaning up...")
        
        # Cleanup recorder
        if hasattr(self, 'recorder'):
            self.recorder.cleanup()
        
        # Cleanup enhanced services
        if ENHANCED_FEATURES_AVAILABLE:
            if hasattr(self, 'document_agent') and self.document_agent:
                self.document_agent.cleanup()
        
        # Display final statistics
        self._display_final_stats()
        
        print("Interview Platform: Goodbye! 👋")
    
    def _display_final_stats(self):
        """Display final statistics before exit."""
        
        print("\n📊 FINAL SESSION STATISTICS")
        print("=" * 50)
        
        if self.stt_service:
            stt_stats = self.stt_service.get_stats()
            if stt_stats["total_requests"] > 0:
                print(f"STT: {stt_stats['total_requests']} requests, {stt_stats['success_rate']:.1%} success")
        
        if self.tts_service:
            tts_stats = self.tts_service.get_stats()
            if tts_stats["total_requests"] > 0:
                print(f"TTS: {tts_stats['total_requests']} requests, {tts_stats['success_rate']:.1%} success")
        
        if ENHANCED_FEATURES_AVAILABLE and hasattr(self, 'document_agent'):
            if self.document_agent:
                doc_stats = self.document_agent.get_stats()
                if doc_stats["total_requests"] > 0:
                    print(f"Documents: {doc_stats['total_requests']} analyzed")
        
        if self.interview_plan:
            print(f"Interview: Plan created with {len(self.interview_plan.interview_sections)} sections")
        
        if self.interview_conductor:
            progress = self.interview_conductor.get_interview_progress()
            print(f"Interview: {progress['interview_status']['completion_percentage']:.1f}% completed")
    
    async def _automatic_interview_setup(self):
        """Automatically set up interview by loading documents from RAG directory."""
        print("\n🚀 AUTOMATIC INTERVIEW SETUP")
        print("=" * 50)
        
        try:
            # Define RAG directory path
            rag_dir = Path(__file__).parent / "RAG"
            
            if not rag_dir.exists():
                print(f"❌ RAG directory not found: {rag_dir}")
                return False
            
            print(f"📁 Loading documents from: {rag_dir}")
            
            # Step 1: Load and analyze resume
            resume_path = rag_dir / "resume.pdf"
            if resume_path.exists():
                print(f"📄 Processing resume: {resume_path}")
                success = await self._auto_process_resume(str(resume_path))
                if not success:
                    return False
            else:
                print(f"❌ Resume not found: {resume_path}")
                return False
            
            # Step 2: Load and analyze job description  
            job_desc_path = rag_dir / "description.txt"
            if job_desc_path.exists():
                print(f"📋 Processing job description: {job_desc_path}")
                success = await self._auto_process_job_description(str(job_desc_path))
                if not success:
                    return False
            else:
                print(f"❌ Job description not found: {job_desc_path}")
                return False
            
            # Step 3: Create interview plan automatically
            print("🎯 Creating interview plan...")
            success = await self._auto_create_interview_plan()
            if not success:
                return False
            
            # Step 4: Start interview automatically
            print("🎬 Starting interview...")
            success = await self._auto_start_interview()
            if not success:
                return False
            
            print("✅ Automatic setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Automatic setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _auto_process_resume(self, file_path: str):
        """Automatically process resume using existing functions."""
        try:
            print("⏳ Analyzing resume...")
            
            if not self.document_agent:
                print("❌ Document intelligence agent not available")
                return False
            
            # Use existing document analysis logic
            analysis_result = await self.document_agent.analyze_resume(file_path)
            
            if analysis_result.success:
                self.resume_analysis = analysis_result.data
                print(f"✅ Resume analysis completed")
                print(f"   📊 Skills identified: {len(self.resume_analysis.technical_skills)}")
                print(f"   💼 Experience years: {self.resume_analysis.total_experience_years}")
                return True
            else:
                print(f"❌ Resume analysis failed: {analysis_result.error}")
                return False
                
        except Exception as e:
            print(f"❌ Resume processing error: {e}")
            return False
    
    async def _auto_process_job_description(self, file_path: str):
        """Automatically process job description using existing functions."""
        try:
            print("⏳ Analyzing job description...")
            
            if not self.document_agent:
                print("❌ Document intelligence agent not available")
                return False
            
            # Use existing document analysis logic
            analysis_result = await self.document_agent.analyze_job_description(file_path)
            
            if analysis_result.success:
                self.job_description_analysis = analysis_result.data
                print(f"✅ Job description analysis completed")
                print(f"   🎯 Role: {self.job_description_analysis.role_title}")
                print(f"   🏢 Company: {self.job_description_analysis.company_name}")
                return True
            else:
                print(f"❌ Job description analysis failed: {analysis_result.error}")
                return False
                
        except Exception as e:
            print(f"❌ Job description processing error: {e}")
            return False
    
    async def _auto_create_interview_plan(self):
        """Automatically create interview plan using existing functions."""
        try:
            if not self.resume_analysis or not self.job_description_analysis:
                print("❌ Missing document analysis data")
                return False
            
            if not self.interview_planning_service:
                print("❌ Interview planning service not available")
                return False
            
            # Use existing interview planning logic
            planning_result = await self.interview_planning_service.create_interview_plan(
                self.resume_analysis,
                self.job_description_analysis
            )
            
            if planning_result.success:
                self.interview_plan = planning_result.data
                print(f"✅ Interview plan created")
                print(f"   📋 Sections: {len(self.interview_plan.interview_sections)}")
                print(f"   ⏱️  Duration: {self.interview_plan.estimated_duration_minutes} minutes")
                return True
            else:
                print(f"❌ Interview plan creation failed: {planning_result.error}")
                return False
                
        except Exception as e:
            print(f"❌ Interview plan creation error: {e}")
            return False
    
    async def _auto_start_interview(self):
        """Automatically start interview using existing functions."""
        try:
            if not self.interview_plan:
                print("❌ Interview plan not available")
                return False
            
            if not self.interview_conductor:
                print("❌ Interview conductor not available")
                return False
            
            # Use existing interview start logic
            start_result = await self.interview_conductor.initialize_interview(
                self.interview_plan,
                {
                    "resume_analysis": self.resume_analysis,
                    "job_description_analysis": self.job_description_analysis
                }
            )
            
            if start_result.success:
                self.interview_active = True
                print("✅ Interview initialized and ready")
                
                # Give opening message
                opening_result = await self.interview_conductor.get_opening_message()
                if opening_result.success:
                    print(f"\n🤖 Interviewer: {opening_result.content}")
                    
                    # Use TTS for opening message
                    if self.tts_service:
                        tts_result = self.tts_service.generate_speech(
                            opening_result.content,
                            play_immediately=True
                        )
                
                return True
            else:
                print(f"❌ Interview initialization failed: {start_result.error}")
                return False
                
        except Exception as e:
            print(f"❌ Interview start error: {e}")
            return False


class EnhancedAudioRecorder:
    """
    Enhanced audio recorder with interview context integration.
    
    This recorder integrates with the interview conductor and provides
    context-aware processing of candidate responses.
    """
    
    def __init__(self, stt_service=None, tts_service=None, document_agent=None, 
                 interview_conductor=None, interview_plan=None):
        # Initialize base recorder functionality
        self.audio = pyaudio.PyAudio()
        self.config = AudioConfig()
        self.stt_config = STTConfig()
        
        # Service references
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.document_agent = document_agent
        self.interview_conductor = interview_conductor
        self.interview_plan = interview_plan
        
        # Recording state
        self.minimal_logging = False
        self.is_recording = False
        self.recording_thread = None
        self.transcription_thread = None
        self.frames = []
        self.current_filename = None
        self.current_filepath = None
        
        # Interview state
        self.interview_turn_count = 0
        
        # Ensure directories exist
        os.makedirs(self.config.RECORDINGS_DIR, exist_ok=True)
        if self.stt_config.SAVE_TRANSCRIPTS:
            os.makedirs(self.stt_config.TRANSCRIPTS_DIR, exist_ok=True)
    
    def check_microphone_access(self):
        """Test microphone access (same as original)."""
        try:
            print("Interview Platform: Checking microphone access...")
            
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"  Device {i}: {device_info['name']}")
            
            # Test recording
            test_stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            
            test_data = test_stream.read(1024, exception_on_overflow=False)
            test_stream.stop_stream()
            test_stream.close()
            
            print("✅ Microphone access confirmed")
            return True
            
        except Exception as e:
            print(f"❌ Microphone access failed - {e}")
            return False
    
    def start_recording(self):
        """Start recording with enhanced logging."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.frames = []
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"interview_recording_{timestamp}.{self.config.OUTPUT_FORMAT}"
        
        if not self.minimal_logging:
            print(f"🎙️  Recording started - {self.current_filename}")
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and start transcription."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if not self.minimal_logging:
            print("🎙️  Recording stopped")
        
        # Save the recording
        self._save_recording()
        
        if self.current_filepath:
            print(f"💾 Recording saved - {self.current_filename}")
            
            # Start transcription
            self._start_transcription()
    
    def _record_audio(self):
        """Record audio data (same as original implementation)."""
        try:
            stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.config.CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    if self.is_recording:  # Only log if we're still supposed to be recording
                        print(f"⚠️  Recording error - {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"💾 Recording setup error - {e}")
            self.is_recording = False
    
    def _save_recording(self):
        """Save recorded audio frames to file."""
        try:
            self.current_filepath = os.path.join(self.config.RECORDINGS_DIR, self.current_filename)
            
            try:
                from pydub import AudioSegment
                
                audio_data = b''.join(self.frames)
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=self.audio.get_sample_size(self.config.FORMAT),
                    frame_rate=self.config.RATE,
                    channels=self.config.CHANNELS
                )
                
                audio_segment.export(
                    self.current_filepath,
                    format=self.config.OUTPUT_FORMAT,
                    bitrate=self.config.MP3_BITRATE,
                    parameters=["-q:a", str(self.config.MP3_QUALITY)]
                )
                
            except ImportError:
                import wave
                with wave.open(self.current_filepath.replace('.mp3', '.wav'), 'wb') as wf:
                    wf.setnchannels(self.config.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.config.FORMAT))
                    wf.setframerate(self.config.RATE)
                    wf.writeframes(b''.join(self.frames))
                self.current_filepath = self.current_filepath.replace('.mp3', '.wav')
                self.current_filename = self.current_filename.replace('.mp3', '.wav')
            
        except Exception as e:
            print(f"💾 Error saving recording - {e}")
            self.current_filepath = None
    
    def _start_transcription(self):
        """Start transcription in a separate thread."""
        if self.transcription_thread and self.transcription_thread.is_alive():
            print("⚠️  Previous transcription still in progress")
            return
        
        if not self.minimal_logging:
            print("🗣️  Starting transcription...")
        
        self.transcription_thread = threading.Thread(target=self._transcribe_and_process)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def _transcribe_and_process(self):
        """Enhanced transcription with interview context processing."""
        try:
            if not self.current_filepath or not os.path.exists(self.current_filepath):
                print("⚠️  No valid audio file for transcription")
                return
            
            # Perform transcription
            result = self.stt_service.transcribe_audio(self.current_filepath)
            
            if result["success"]:
                transcription_text = result["text"]
                processing_time = result["processing_time"]
                
                if not self.minimal_logging:
                    print(f"✅ Transcription completed in {processing_time:.2f}s")
                
                print(f"🗣️  You: {transcription_text}")
                
                # Enhanced processing with interview conductor
                if self.interview_conductor and self.interview_plan:
                    import asyncio
                    asyncio.create_task(self._process_with_interview_conductor(transcription_text))
                else:
                    # Fallback to basic TTS response
                    if self.tts_service:
                        response_text = f"I heard you say: {transcription_text}"
                        tts_result = self.tts_service.generate_speech(
                            response_text, 
                            play_immediately=True
                        )
                
                # Save transcript
                if self.stt_config.SAVE_TRANSCRIPTS:
                    self._save_transcript(transcription_text, result)
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Transcription failed - {error_msg}")
                
        except Exception as e:
            print(f"❌ Transcription error - {e}")

    async def _process_with_interview_conductor(self, transcription_text: str):
        """Process candidate response with interview conductor."""
        try:
            print("🤖 Processing response with AI Interview Conductor...")
            
            # Increment turn count
            self.interview_turn_count += 1
            
            # Create context for this interview turn
            additional_context = {
                "turn_number": self.interview_turn_count,
                "audio_file": self.current_filename,
                "transcription_confidence": "high",  # Could be extracted from STT result
                "response_length": len(transcription_text.split()),
                "recording_timestamp": datetime.now().isoformat()
            }
            
            # Process with interview conductor
            conductor_response = await self.interview_conductor.conduct_interview_turn(
                transcription_text,
                additional_context
            )
            
            if conductor_response.success:
                print(f"🤖 Interviewer: {conductor_response.content}")
                
                # Use TTS to speak the interviewer's response
                if self.tts_service:
                    print("🔊 Speaking interviewer response...")
                    tts_result = self.tts_service.generate_speech(
                        conductor_response.content,
                        play_immediately=True
                    )
                    
                    if not tts_result["success"]:
                        print(f"⚠️  TTS failed: {tts_result.get('error', 'Unknown error')}")
                
                # Show interview progress
                progress = self.interview_conductor.get_interview_progress()
                if not self.minimal_logging:
                    print(f"📊 Progress: {progress['interview_status']['completion_percentage']:.1f}% | "
                          f"Phase: {progress['current_phase']} | "
                          f"Questions: {progress['question_progress']['questions_asked']}")
                
                # Check if interview is complete
                if not progress['interview_status']['should_continue']:
                    print("🎉 Interview completed!")
                    await self._handle_interview_completion()
                    
            else:
                print(f"❌ Interview conductor error: {conductor_response.error}")
                
        except Exception as e:
            print(f"❌ Interview conductor processing error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_interview_completion(self):
        """Handle interview completion and generate summary."""
        try:
            print("\n🎉 INTERVIEW COMPLETED!")
            print("=" * 50)
            
            # Generate interview summary
            summary = self.interview_conductor.generate_interview_summary()
            
            print(f"📊 Interview Duration: {summary['interview_metadata']['duration_minutes']} minutes")
            print(f"❓ Questions Asked: {summary['interview_metadata']['questions_asked']}")
            print(f"📝 Responses Evaluated: {summary['performance_summary']['total_responses']}")
            print(f"📈 Performance Trend: {summary['performance_summary']['performance_trend']}")
            
            print(f"\n✅ {summary['recommendations']['overall_assessment']}")
            
            # Optionally save full summary
            summary_path = os.path.join(
                self.config.RECORDINGS_DIR.parent, 
                f"interview_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(summary_path, 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Summary saved: {summary_path}")
            
        except Exception as e:
            print(f"❌ Interview completion error: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_transcript(self, transcription_text: str, stt_result: dict):
        """Save transcription to file."""
        try:
            if not self.stt_config.SAVE_TRANSCRIPTS:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_filename = f"transcript_{timestamp}.txt"
            transcript_path = os.path.join(self.stt_config.TRANSCRIPTS_DIR, transcript_filename)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Processing Time: {stt_result.get('processing_time', 'N/A')}s\n")
                f.write(f"Audio File: {self.current_filename}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Transcription:\n{transcription_text}\n")
                
        except Exception as e:
            print(f"💾 Error saving transcript - {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()
        
        if self.tts_service:
            try:
                self.tts_service.cleanup()
            except Exception as e:
                print(f"🧹 TTS cleanup error - {e}")


def main():
    """Entry point for the enhanced application."""
    try:
        # Create and run enhanced platform
        app = EnhancedInterviewPlatform(minimal_logging=False)
        asyncio.run(app.run())
    except Exception as e:
        print(f"❌ Enhanced Interview Platform failed to start - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

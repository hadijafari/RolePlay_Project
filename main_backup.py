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
    
    New Workflow:
    1. Document Upload & Analysis Phase
    2. Interview Plan Generation  
    3. Context-Aware Interview Conduction
    4. Real-time Interview Management
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
        
        # Initialize audio recorder
        self.recorder = EnhancedAudioRecorder(
            stt_service=self.stt_service,
            tts_service=self.tts_service,
            document_agent=getattr(self, 'document_agent', None),
            interview_conductor=None,  # Will be set after interview plan is created
            interview_plan=None
        )
        self.recorder.minimal_logging = self.minimal_logging
        self.running = False
    
    def _initialize_core_services(self):
        """Initialize core STT/TTS services."""
        
        # Initialize STT service
        self.stt_service = None
        stt_config = STTConfig()
        
        if stt_config.ENABLED:
            try:
                self.stt_service = STTService()
                print("Interview Platform: STT service initialized")
            except Exception as e:
                print(f"Interview Platform: STT service initialization failed - {e}")
        
        # Initialize TTS service
        self.tts_service = None
        try:
            self.tts_service = TTSService()
            print("Interview Platform: TTS service initialized")
        except Exception as e:
            print(f"Interview Platform: TTS service initialization failed - {e}")
    
    def _initialize_enhanced_services(self):
        """Initialize enhanced document analysis and interview planning services."""
        
        try:
            # Document Intelligence Agent
            self.document_agent = DocumentIntelligenceAgent("Document Analyzer")
            print("Interview Platform: Document Intelligence Agent initialized")
            
            # Interview Planning Service  
            self.interview_planning_service = InterviewPlanningService()
            print("Interview Platform: Interview Planning Service initialized")
            
            # Context Injection Service
            self.context_service = ContextInjectionService()
            print("Interview Platform: Context Injection Service initialized")
            
        except Exception as e:
            print(f"Interview Platform: Enhanced services initialization failed - {e}")
            self.document_agent = None
            self.interview_planning_service = None
            self.context_service = None
    
    def display_welcome(self):
        """Display enhanced welcome message."""
        
        print("=" * 80)
        print("🎤 ENHANCED AI Interview Platform - Advanced Interview Management")
        
        if ENHANCED_FEATURES_AVAILABLE:
            print("📄 Document Analysis: Resume & Job Description Processing")
            print("🧠 Interview Planning: AI-Generated Strategic Interview Plans")
            print("🎯 Context-Aware Interviewing: Dynamic Question Selection")
        
        if self.stt_service:
            print("🗣️  Speech-to-Text: Real-time transcription")
        if self.tts_service:
            print("🔊 Text-to-Speech: AI voice responses")
        
        print("=" * 80)
        
        if ENHANCED_FEATURES_AVAILABLE:
            print("\n📋 ENHANCED INTERVIEW WORKFLOW:")
            print("1. DOCUMENT UPLOAD: Upload resume and job description")
            print("2. ANALYSIS PHASE: AI analyzes documents and creates interview plan")  
            print("3. INTERVIEW MODE: Conduct structured interview with AI guidance")
            print("4. RECORDING: Hold SPACEBAR to record responses")
            print("5. AI FEEDBACK: Real-time question suggestions and evaluation")
            print("\nCOMMANDS:")
            print("  'upload resume' - Upload and analyze resume")
            print("  'upload job' - Upload and analyze job description")
            print("  'create plan' - Generate interview plan")
            print("  'start interview' - Begin structured interview")
            print("  'show progress' - View interview progress")
            print("  'show stats' - Display system statistics")
        
        print("\nAUDIO CONTROLS:")
        print("  SPACEBAR (hold): Record audio response")
        print("  ESC or Ctrl+C: Exit application")
        
        print("\nRECORDING & ANALYSIS:")
        print("- Audio saved in 'recordings/' directory with timestamps")
        if self.stt_service:
            stt_config = STTConfig()
            print(f"- Auto-transcription with {stt_config.MODEL} model")
            print(f"- Target processing time: {stt_config.TARGET_LATENCY}s")
        
        print("\n" + "=" * 80)
        
        if ENHANCED_FEATURES_AVAILABLE:
            print("💡 TIP: Start by uploading resume and job description for full AI interview experience!")
        else:
            print("📝 NOTE: Running in basic mode - upload enhanced modules for full features")
        
        print("-" * 80)
    
    # Command input handling removed - now using automatic setup
    
    async def _handle_resume_upload(self):
        """Handle resume upload and analysis."""
        
        print("\n📄 RESUME UPLOAD & ANALYSIS")
        print("Supported formats: PDF, DOCX, TXT")
        
        file_path = input("Enter resume file path: ").strip().strip('"').strip("'")
        
        if not file_path:
            print("❌ No file path provided")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"🔍 Analyzing resume: {Path(file_path).name}")
        print("⏳ This may take 30-60 seconds for comprehensive analysis...")
        
        try:
            # Analyze resume
            result = await self.document_agent.analyze_resume(file_path)
            
            if result.success:
                print(f"✅ Resume analysis completed in {result.processing_time:.2f}s")
                print(f"📊 Analysis confidence: {result.metadata.get('confidence_score', 0.8):.1%}")
                print("\n📋 RESUME ANALYSIS SUMMARY:")
                print("=" * 50)
                print(result.content[:800] + "..." if len(result.content) > 800 else result.content)
                
                # Store analysis
                self.current_resume_path = file_path
                self.resume_analysis = ResumeAnalysis(**result.metadata['resume_analysis'])
                
                print(f"\n✅ Resume analysis saved. Ready for interview planning!")
                
                # Suggest next steps
                if not self.job_description_analysis:
                    print("💡 Next step: Upload job description with 'upload job'")
                else:
                    print("💡 Next step: Create interview plan with 'create plan'")
            else:
                print(f"❌ Resume analysis failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Resume analysis error: {e}")
    
    async def _handle_job_description_upload(self):
        """Handle job description upload and analysis."""
        
        print("\n📋 JOB DESCRIPTION UPLOAD & ANALYSIS") 
        print("Supported formats: PDF, DOCX, TXT")
        
        file_path = input("Enter job description file path: ").strip().strip('"').strip("'")
        
        if not file_path:
            print("❌ No file path provided")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"🔍 Analyzing job description: {Path(file_path).name}")
        print("⏳ This may take 30-60 seconds for comprehensive analysis...")
        
        try:
            # Analyze job description
            result = await self.document_agent.analyze_job_description(file_path)
            
            if result.success:
                print(f"✅ Job description analysis completed in {result.processing_time:.2f}s")
                print(f"📊 Analysis confidence: {result.metadata.get('confidence_score', 0.8):.1%}")
                print("\n📋 JOB DESCRIPTION ANALYSIS SUMMARY:")
                print("=" * 50)
                print(result.content[:800] + "..." if len(result.content) > 800 else result.content)
                
                # Store analysis
                self.current_job_description_path = file_path
                self.job_description_analysis = JobDescriptionAnalysis(**result.metadata['job_analysis'])
                
                print(f"\n✅ Job description analysis saved. Ready for interview planning!")
                
                # Suggest next steps
                if not self.resume_analysis:
                    print("💡 Next step: Upload resume with 'upload resume'")
                else:
                    print("💡 Next step: Create interview plan with 'create plan'")
            else:
                print(f"❌ Job description analysis failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Job description analysis error: {e}")
    
    async def _handle_interview_plan_creation(self):
        """Handle interview plan creation."""
        
        if not self.resume_analysis or not self.job_description_analysis:
            print("❌ Both resume and job description must be analyzed first")
            print("   Use 'upload resume' and 'upload job' commands")
            return
        
        print("\n🎯 CREATING INTERVIEW PLAN")
        print("⏳ Generating strategic interview plan based on document analysis...")
        
        try:
            # Create candidate match analysis
            print("🔍 Analyzing candidate-job match...")
            match_result = await self.document_agent.create_candidate_match_analysis()
            
            if match_result.success:
                self.candidate_match = CandidateMatch(**match_result.metadata['candidate_match'])
                print(f"✅ Match analysis completed - {match_result.metadata['match_score']:.1%} overall match")
            
            # Generate interview plan
            print("📝 Generating comprehensive interview plan...")
            plan_result = await self.interview_planning_service.generate_interview_plan(
                self.resume_analysis,
                self.job_description_analysis,
                self.candidate_match
            )
            
            if plan_result.success:
                self.interview_plan = InterviewPlan(**plan_result.result_data)
                
                print(f"✅ Interview plan created in {plan_result.processing_time_seconds:.2f}s")
                print(f"📊 Plan contains {plan_result.metadata['total_sections']} sections with {plan_result.metadata['total_questions']} questions")
                print(f"⏱️  Estimated duration: {self.interview_plan.total_estimated_duration_minutes} minutes")
                
                print("\n📋 INTERVIEW PLAN SUMMARY:")
                print("=" * 50)
                
                # Display interview sections
                for i, section in enumerate(self.interview_plan.interview_sections, 1):
                    print(f"{i}. {section.section_name} ({section.estimated_duration_minutes} min)")
                    print(f"   Phase: {section.phase.value.title()}")
                    print(f"   Questions: {len(section.questions)}")
                    if section.objectives:
                        print(f"   Objectives: {', '.join(section.objectives[:2])}...")
                
                print(f"\n🎯 Key Focus Areas: {', '.join(self.interview_plan.key_focus_areas[:3])}...")
                print(f"📊 Evaluation Priorities: {', '.join(self.interview_plan.evaluation_priorities[:2])}...")
                
                print("\n✅ Interview plan ready!")
                print("💡 Next step: Start interview with 'start interview'")
                
            else:
                print(f"❌ Interview plan creation failed: {plan_result.error_message}")
                
        except Exception as e:
            print(f"❌ Interview plan creation error: {e}")
    
    async def _handle_interview_start(self):
        """Handle interview start with context-aware conductor."""
        
        if not self.interview_plan:
            print("❌ Interview plan must be created first")
            print("   Use 'create plan' command after uploading documents")
            return
        
        print("\n🎬 STARTING STRUCTURED INTERVIEW")
        print("=" * 50)
        
        try:
            # Create interview conductor agent with the interview plan
            print("🤖 Initializing AI Interview Conductor...")
            self.interview_conductor = InterviewConductorAgent(
                self.interview_plan,
                "AI Interview Conductor"
            )
            
            # Update recorder with interview conductor
            self.recorder.interview_conductor = self.interview_conductor
            self.recorder.interview_plan = self.interview_plan
            
            # Set interview as active
            self.interview_active = True
            
            print("✅ Interview conductor initialized with structured plan")
            print(f"📋 Interview plan: {self.interview_plan.plan_id}")
            print(f"⏱️  Estimated duration: {self.interview_plan.total_estimated_duration_minutes} minutes")
            
            # Display interview progress
            progress = self.interview_conductor.get_interview_progress()
            print(f"📊 Current section: {progress['section_progress']['current_section']}/{progress['section_progress']['total_sections']}")
            print(f"🎯 Current phase: {progress['current_phase']}")
            
            print("\n🎤 INTERVIEW MODE ACTIVE")
            print("Hold SPACEBAR to record your responses")
            print("The AI interviewer will provide questions and feedback")
            print("Type 'show progress' to see interview status")
            print("-" * 50)
            
            # Give initial interview question
            initial_response = await self.interview_conductor.conduct_interview_turn(
                "Hello, I'm ready to begin the interview. Please start with your opening question."
            )
            
            if initial_response.success:
                print(f"\n🤖 Interviewer: {initial_response.content}")
                
                # Use TTS to speak the question if available
                if self.tts_service:
                    print("🔊 Speaking question...")
                    tts_result = self.tts_service.generate_speech(
                        initial_response.content, 
                        play_immediately=True
                    )
                    if not tts_result["success"]:
                        print(f"⚠️  TTS failed: {tts_result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Failed to start interview: {initial_response.error}")
                self.interview_active = False
                
        except Exception as e:
            print(f"❌ Interview start error: {e}")
            self.interview_active = False
    
    def _show_interview_progress(self):
        """Show current interview progress."""
        
        if not self.interview_conductor:
            print("❌ No active interview")
            return
        
        try:
            progress = self.interview_conductor.get_interview_progress()
            
            print("\n📊 INTERVIEW PROGRESS")
            print("=" * 50)
            print(f"Interview ID: {progress['interview_id']}")
            print(f"Current Phase: {progress['current_phase'].title()}")
            print(f"Section Progress: {progress['section_progress']['current_section']}/{progress['section_progress']['total_sections']}")
            print(f"Questions Asked: {progress['question_progress']['questions_asked']}")
            print(f"Time Remaining: {progress['time_management']['time_remaining_minutes']} minutes")
            print(f"Completion: {progress['interview_status']['completion_percentage']:.1f}%")
            print(f"Status: {'Active' if progress['interview_status']['should_continue'] else 'Completed'}")
            
            if progress['performance_indicators']['responses_evaluated'] > 0:
                print(f"Responses Evaluated: {progress['performance_indicators']['responses_evaluated']}")
                print(f"Performance Trend: {progress['performance_indicators']['performance_trend']}")
            
        except Exception as e:
            print(f"❌ Error showing progress: {e}")
    
    def _show_system_stats(self):
        """Show system statistics."""
        
        print("\n📈 SYSTEM STATISTICS")
        print("=" * 50)
        
        # STT Stats
        if self.stt_service:
            stt_stats = self.stt_service.get_stats()
            print(f"STT Requests: {stt_stats['total_requests']}")
            print(f"STT Success Rate: {stt_stats['success_rate']:.1%}")
            print(f"STT Avg Time: {stt_stats['average_processing_time']:.2f}s")
        
        # TTS Stats  
        if self.tts_service:
            tts_stats = self.tts_service.get_stats()
            print(f"TTS Requests: {tts_stats['total_requests']}")
            print(f"TTS Success Rate: {tts_stats['success_rate']:.1%}")
            print(f"TTS Cache Hits: {tts_stats['cache_hits']}")
        
        # Enhanced service stats
        if ENHANCED_FEATURES_AVAILABLE:
            if hasattr(self, 'document_agent') and self.document_agent:
                doc_stats = self.document_agent.get_stats()
                print(f"Agent Requests: {doc_stats['total_requests']}")
                print(f"Agent Success Rate: {doc_stats['success_rate']:.1%}")
                
            if hasattr(self, 'interview_planning_service') and self.interview_planning_service:
                planning_stats = self.interview_planning_service.get_service_stats()
                print(f"Interview Plans: {planning_stats['plans_generated']}")
                print(f"Planning Success Rate: {planning_stats['success_rate']:.1%}")
        
        print(f"Interview Active: {'Yes' if self.interview_active else 'No'}")
        print(f"Documents Analyzed: {'Resume + Job' if self.resume_analysis and self.job_description_analysis else 'None'}")
    
    def _show_document_analysis(self):
        """Show document analysis results."""
        
        if not self.resume_analysis and not self.job_description_analysis:
            print("❌ No documents analyzed yet")
            return
        
        print("\n📄 DOCUMENT ANALYSIS RESULTS")
        print("=" * 50)
        
        if self.resume_analysis:
            print("✅ Resume Analysis Available")
            print(f"   Candidate: {self.resume_analysis.personal_info.full_name or 'Name not extracted'}")
            print(f"   Technical Skills: {len(self.resume_analysis.technical_skills)}")
            print(f"   Work Experience: {len(self.resume_analysis.work_experience)} positions")
            print(f"   Confidence: {self.resume_analysis.confidence_score:.1%}")
        
        if self.job_description_analysis:
            print("✅ Job Description Analysis Available")
            print(f"   Position: {self.job_description_analysis.job_title}")
            print(f"   Company: {self.job_description_analysis.company_info.company_name or 'Not specified'}")
            print(f"   Seniority: {self.job_description_analysis.role_seniority}")
            print(f"   Confidence: {self.job_description_analysis.confidence_score:.1%}")
        
        if self.candidate_match:
            print("✅ Candidate Match Analysis Available")
            print(f"   Overall Match: {self.candidate_match.overall_match_score:.1%}")
            print(f"   Strong Matches: {len(self.candidate_match.strong_matches)}")
            print(f"   Skill Gaps: {len(self.candidate_match.skill_gaps)}")
        
        if self.interview_plan:
            print("✅ Interview Plan Available")
            print(f"   Plan ID: {self.interview_plan.plan_id}")
            print(f"   Sections: {len(self.interview_plan.interview_sections)}")
            print(f"   Duration: {self.interview_plan.total_estimated_duration_minutes} minutes")
    
    def _clear_interview_data(self):
        """Clear all interview data."""
        
        confirm = input("⚠️  Clear all interview data? (yes/no): ").strip().lower()
        if confirm == 'yes':
            self.current_resume_path = None
            self.current_job_description_path = None
            self.resume_analysis = None
            self.job_description_analysis = None
            self.candidate_match = None
            self.interview_plan = None
            self.interview_active = False
            self.interview_conductor = None
            
            if hasattr(self, 'document_agent') and self.document_agent:
                self.document_agent.clear_analysis_data()
            
            print("✅ All interview data cleared")
        else:
            print("❌ Data clear cancelled")
    
    def _show_help(self):
        """Show available commands."""
        
        print("\n📚 AVAILABLE COMMANDS")
        print("=" * 50)
        print("📄 Document Management:")
        print("  upload resume    - Upload and analyze resume file")
        print("  upload job       - Upload and analyze job description")
        print("  show analysis    - Show document analysis results")
        print("  clear data       - Clear all analysis data")
        print("\n🎯 Interview Management:")
        print("  create plan      - Generate interview plan")
        print("  start interview  - Begin structured interview")
        print("  show progress    - View interview progress")
        print("\n📊 System Information:")
        print("  show stats       - Display system statistics")
        print("  help             - Show this help message")
        print("  exit/quit        - Exit application")
        print("\n🎤 Audio Controls:")
        print("  SPACEBAR (hold)  - Record audio response")
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
                # Run enhanced mode with command handling
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
        
        # Test TTS
        if self.tts_service and self.tts_service.test_connection():
            print("✅ TTS service ready")
        elif self.tts_service:
            print("⚠️  TTS service connection issues")
        
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
            
            test_stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            test_stream.close()
            print("✓ Microphone access confirmed")
            return True
            
        except Exception as e:
            print(f"✗ Microphone access error: {e}")
            return False
    
    # [Include all original recording methods: generate_filename, start_recording, 
    # stop_recording, _record_audio, _save_recording with same implementation]
    
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
        
        if not self.minimal_logging:
            print(f"🎤 Recording started - {self.current_filename}")
        
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop audio recording and process."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.frames:
            self._save_recording()
            if not self.minimal_logging:
                print(f"💾 Recording saved - {self.current_filename}")
            
            if (self.stt_service and self.stt_config.ENABLED and 
                self.stt_config.AUTO_TRANSCRIBE and self.current_filepath):
                self._start_transcription()
        else:
            print("⚠️  No audio data recorded")
    
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
            print(f"🎤 Recording error - {e}")
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
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Full summary saved: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error generating interview summary: {e}")
    
    def _save_transcript(self, text: str, result: dict):
        """Save transcription to file (same as original implementation)."""
        try:
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
                    "model": result.get("model", "whisper-1"),
                    "interview_turn": getattr(self, 'interview_turn_count', 0)
                }
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            else:
                transcript_filename = f"{audio_name}_transcript.txt"
                transcript_path = os.path.join(self.stt_config.TRANSCRIPTS_DIR, transcript_filename)
                
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(f"Audio File: {self.current_filename}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Interview Turn: {getattr(self, 'interview_turn_count', 0)}\n")
                    f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
                    f.write(f"Language: {result.get('language', 'unknown')}\n")
                    f.write("-" * 50 + "\n")
                    f.write(text)
            
            if not self.minimal_logging:
                print(f"📄 Transcript saved - {transcript_filename}")
                
        except Exception as e:
            print(f"📄 Error saving transcript - {e}")
    
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
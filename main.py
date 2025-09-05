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

Tab-controlled audio recording with comprehensive interview management.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
import json

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
        # print(f"main.py: Class EnhancedInterviewPlatform.__init__ called: Initialize the enhanced interview platform")
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
            interview_conductor=None,  # Explicitly None for now
            interview_plan=None  # Explicitly None for now
        )
        
        # Application state
        self.running = False
    
    def update_recorder_interview_components(self):
        """Update recorder with interview components after they're created."""
        # print(f"main.py: Class EnhancedInterviewPlatform.update_recorder_interview_components called: Update recorder with interview components after they're created.")
        if hasattr(self, 'recorder'):
            self.recorder.interview_conductor = self.interview_conductor
            self.recorder.interview_plan = self.interview_plan
            print("✅ Recorder updated with interview conductor and plan")
    
    def _initialize_core_services(self):
        """Initialize core STT and TTS services."""

        # print(f"main.py: Class EnhancedInterviewPlatform._initialize_core_services called: Initialize core STT and TTS services.")
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

        # print(f"main.py: Class EnhancedInterviewPlatform._initialize_enhanced_services called: Initialize enhanced AI services for document analysis and interview planning.")
        try:
            # Initialize document intelligence agent
            print("Initialize document intelligence agent")
            self.document_agent = DocumentIntelligenceAgent()
            if not self.minimal_logging:
                print("✅ Document Intelligence Agent initialized")
        except Exception as e:
            print(f"⚠️  Document Intelligence Agent initialization failed: {e}")
            self.document_agent = None
        
        try:
            # Initialize interview planning service
            print("Initialize interview planning service")
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
        

        self.interview_conductor = None
    
    def display_welcome(self):
        """Display enhanced welcome message with feature overview."""

        # print(f"main.py: Class EnhancedInterviewPlatform.display_welcome called: Display enhanced welcome message with feature overview.")
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
            print("   • TAB (hold): Record your response")
            print("   • Ctrl+C: Exit interview")
            print()
        else:
            print("📝 NOTE: Running in basic mode - upload enhanced modules for full features")
        
        print("-" * 80)
    
    
    def setup_keyboard_handlers(self):
        """Set up keyboard event handlers."""

        # print(f"main.py: Class EnhancedInterviewPlatform.setup_keyboard_handlers called: Set up keyboard event handlers.")
        # Tab key handlers for audio recording
        keyboard.on_press_key('tab', self._on_tab_press)
        keyboard.on_release_key('tab', self._on_tab_release)
        
        # Ctrl+C handler for exit (handled by KeyboardInterrupt in main loop)
    
    def _on_tab_press(self, event):
        """Handle tab key press for recording."""   
        if not self.recorder.is_recording:
            self.recorder.start_recording()
    
    def _on_tab_release(self, event):
        """Handle tab key release to stop recording."""
        if self.recorder.is_recording:
            self.recorder.stop_recording()
    
    async def run(self):
        """Main application loop with enhanced features."""
        # print(f"main.py: Class EnhancedInterviewPlatform.run called: Main application loop with enhanced features.")
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

        # print(f"main.py: Class EnhancedInterviewPlatform._run_enhanced_mode called: Run application in enhanced mode with automatic setup.")
        # Automatically set up the interview
        setup_success = await self._automatic_interview_setup()
        
        if not setup_success:
            print("❌ Failed to set up interview automatically. Exiting...")
            self.running = False
            return
        
        # Main loop for keyboard events only (no command input)
        print("\n🎤 Interview ready! Hold TAB to record, press Ctrl+C to exit")
        try:
            while self.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterview Platform: Interrupted by user")
            self.running = False
        except Exception as e:
            print(f"Interview Platform: Unexpected error - {e}")
            self.running = False
    
    async def _run_basic_mode(self):
        """Run application in basic mode (original functionality)."""

        # print(f"main.py: Class EnhancedInterviewPlatform._run_basic_mode called: Run application in basic mode (original functionality).")
        print("\n📝 Basic Mode: Audio recording and transcription only")
        print("Hold TAB to record, release to stop and transcribe")
        print("Press Ctrl+C to exit")
        
        while self.running:
            await asyncio.sleep(0.1)
    
    async def _cleanup(self):
        """Clean up all resources."""
        # print(f"main.py: Class EnhancedInterviewPlatform._cleanup called: Clean up all resources.")
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

        # print(f"main.py: Class EnhancedInterviewPlatform._display_final_stats called: Display final statistics before exit.")
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
    
    def _find_file_with_extensions(self, directory: Path, base_name: str, extensions: list) -> Path:
        """Find a file with the given base name and any of the supported extensions."""
        # print(f"main.py: Class EnhancedInterviewPlatform._find_file_with_extensions called: Find a file with the given base name and any of the supported extensions.")
        for ext in extensions:
            file_path = directory / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
        return None

    async def _automatic_interview_setup(self):
        """Automatically set up interview by loading documents from RAG directory."""

        # print(f"main.py: Class EnhancedInterviewPlatform._automatic_interview_setup called: Automatically set up interview by loading documents from RAG directory.")
        print("\n🚀 AUTOMATIC INTERVIEW SETUP")
        print("=" * 50)
        
        try:
            # Define RAG directory path
            rag_dir = Path(__file__).parent / "RAG"
            
            if not rag_dir.exists():
                print(f"❌ RAG directory not found: {rag_dir}")
                return False
            
            print(f"📁 Loading documents from: {rag_dir}")
            
            # Supported file extensions
            supported_extensions = ['.pdf', '.docx', '.txt']
            
            # Step 1: Load and analyze resume
            resume_path = self._find_file_with_extensions(rag_dir, "resume", supported_extensions)
            if resume_path:
                print(f"📄 Processing resume: {resume_path}")
                success = await self._auto_process_resume(str(resume_path))
                if not success:
                    return False
            else:
                print(f"❌ Resume not found. Looking for: resume.pdf, resume.docx, or resume.txt in {rag_dir}")
                return False
            
            # Step 2: Load and analyze job description  
            job_desc_path = self._find_file_with_extensions(rag_dir, "description", supported_extensions)
            if job_desc_path:
                print(f"📋 Processing job description: {job_desc_path}")
                success = await self._auto_process_job_description(str(job_desc_path))
                if not success:
                    return False
            else:
                print(f"❌ Job description not found. Looking for: description.pdf, description.docx, or description.txt in {rag_dir}")
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

        # print(f"main.py: Class EnhancedInterviewPlatform._auto_process_resume called: Automatically process resume using existing functions.")
        try:
            print("⏳ Analyzing resume...")
            
            if not self.document_agent:
                print("❌ Document intelligence agent not available")
                return False
            
            # Use existing document analysis logic
            analysis_result = await self.document_agent.analyze_resume(file_path)
            
            if analysis_result.success:
                # Extract resume analysis from metadata - works for all document types
                print(f"In _auto_process_resume method of EnhancedInterviewPlatform: Almost done with resume analysis")
                if hasattr(analysis_result, 'metadata') and 'resume_analysis' in analysis_result.metadata:
                    # Get from metadata (preferred)
                    resume_data = analysis_result.metadata['resume_analysis']
                elif hasattr(analysis_result, 'data'):
                    # Fallback to data field if it exists
                    resume_data = analysis_result.data
                else:
                    # Last resort: try to extract from content or other fields
                    print("⚠️  Resume analysis data not found in expected location")
                    return False
                
                # Create ResumeAnalysis object from the data
                try:
                    from models.interview_models import ResumeAnalysis
                    self.resume_analysis = ResumeAnalysis(**resume_data)
                except Exception as e:
                    print(f"❌ Failed to create ResumeAnalysis object: {e}")
                    return False
                
                print(f"✅ Resume analysis completed")
                print(f"   📊 Skills identified: {len(self.resume_analysis.technical_skills)}")
                # the file name should be resume_analysis_date
                # with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\resume_analysis_{self.resume_analysis.analysis_timestamp.strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                #     f.write(json.dumps(self.resume_analysis.model_dump(mode='json'), indent=2))
                # print(f"In _auto_process_resume method of EnhancedInterviewPlatform: Resume analysis saved to file")


                # Safely access experience years - works for all document types
                try:
                    if self.resume_analysis.career_progression:
                        experience_years = self.resume_analysis.career_progression.total_experience_years
                        print(f"   💼 Experience years: {experience_years}")
                    else:
                        print(f"   💼 Experience years: Not specified")
                except AttributeError:
                    print(f"   💼 Experience years: Not available")
                
                return True
            else:
                # Handle different error field names - works for all document types
                error_msg = getattr(analysis_result, 'error_message', None) or getattr(analysis_result, 'error', 'Unknown error')
                print(f"❌ Resume analysis failed: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Resume processing error: {e}")
            return False
    
    async def _auto_process_job_description(self, file_path: str):
        """Automatically process job description using existing functions."""

        # print(f"main.py: Class EnhancedInterviewPlatform._auto_process_job_description called: Automatically process job description using existing functions.")
        try:
            print("⏳ Analyzing job description...")
            
            if not self.document_agent:
                print("❌ Document intelligence agent not available")
                return False
            
            # Use existing document analysis logic
            analysis_result = await self.document_agent.analyze_job_description(file_path)
            
            if analysis_result.success:
                # Extract job description analysis from metadata - works for all document types
                if hasattr(analysis_result, 'metadata') and 'job_analysis' in analysis_result.metadata:
                    # Get from metadata (preferred)
                    job_data = analysis_result.metadata['job_analysis']
                elif hasattr(analysis_result, 'data'):
                    # Fallback to data field if it exists
                    job_data = analysis_result.data
                else:
                    # Last resort: try to extract from content or other fields
                    print("⚠️  Job description analysis data not found in expected location")
                    return False
                
                # Create JobDescriptionAnalysis object from the data
                try:
                    from models.interview_models import JobDescriptionAnalysis
                    self.job_description_analysis = JobDescriptionAnalysis(**job_data)
                except Exception as e:
                    print(f"❌ Failed to create JobDescriptionAnalysis object: {e}")
                    return False
                
                print(f"✅ Job description analysis completed")
                
                # Safely access job title - works for all document types
                try:
                    job_title = self.job_description_analysis.job_title
                    print(f"   🎯 Role: {job_title}")
                except AttributeError:
                    print(f"   🎯 Role: Not specified")
                
                # Safely access company info - works for all document types
                try:
                    if self.job_description_analysis.company_info and self.job_description_analysis.company_info.company_name:
                        company_name = self.job_description_analysis.company_info.company_name
                        print(f"   🏢 Company: {company_name}")
                        with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\job_description_analysis_{self.job_description_analysis.analysis_timestamp.strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                            f.write(json.dumps(self.job_description_analysis.model_dump(mode='json'), indent=2))
                        print(f"In _auto_process_job_description method of EnhancedInterviewPlatform: Job description analysis saved to file")
                    else:
                        print(f"   🏢 Company: Not specified")
                except AttributeError:
                    print(f"   🏢 Company: Not available")
                
                return True
            else:
                # Handle different error field names - works for all document types
                error_msg = getattr(analysis_result, 'error_message', None) or getattr(analysis_result, 'error', 'Unknown error')
                print(f"❌ Job description analysis failed: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Job description processing error: {e}")
            return False
    
    async def _auto_create_interview_plan(self):
        """Automatically create interview plan using existing functions."""

        # print(f"main.py: Class EnhancedInterviewPlatform._auto_create_interview_plan called: Automatically create interview plan using existing functions.")
        try:
            if not self.resume_analysis or not self.job_description_analysis:
                print("❌ Missing document analysis data")
                return False
            
            if not self.interview_planning_service:
                print("❌ Interview planning service not available")
                return False
            
            # Use existing interview planning logic
            planning_result = await self.interview_planning_service.generate_interview_plan(
                self.resume_analysis,
                self.job_description_analysis
            )
            
            
            if planning_result.success:
                # Extract interview plan from result_data - works for all document types
                if hasattr(planning_result, 'result_data'):
                    # Get from result_data (preferred)
                    plan_data = planning_result.result_data
                elif hasattr(planning_result, 'data'):
                    # Fallback to data field if it exists
                    plan_data = planning_result.data
                else:
                    # Last resort: try to extract from content or other fields
                    print("⚠️  Interview plan data not found in expected location")
                    return False
                
                # Create InterviewPlan object from the data
                try:
                    from models.interview_models import InterviewPlan
                    self.interview_plan = InterviewPlan(**plan_data)
                    # save to file
                    with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w", encoding="utf-8") as f:
                        f.write(json.dumps(self.interview_plan.model_dump(mode='json'), indent=2, default=str))                          
                    print(f"In _auto_create_interview_plan method of EnhancedInterviewPlatform: Interview plan saved to file")
                except Exception as e:
                    print(f"❌ Failed to create InterviewPlan object: {e}")
                    return False
                
                print(f"✅ Interview plan created")
                
                # Safely access sections - works for all document types
                try:
                    sections_count = len(self.interview_plan.interview_sections)
                    print(f"   📋 Sections: {sections_count}")
                except AttributeError:
                    print(f"   📋 Sections: Not available")
                
                # Safely access duration - works for all document types
                try:
                    duration = self.interview_plan.total_estimated_duration_minutes
                    print(f"   ⏱️  Duration: {duration} minutes")
                except AttributeError:
                    print(f"   ⏱️  Duration: Not specified")
                
                # Print detailed interview plan
                # print("\n" + "="*80)
                # print("📋 DETAILED INTERVIEW PLAN")
                # print("="*80)
                # self._print_detailed_interview_plan()
                # print("="*80)
                
                # Initialize interview conductor now that we have the plan
                try:
                    from agents.interview_conductor_agent import InterviewConductorAgent
                    self.interview_conductor = InterviewConductorAgent(interview_plan=self.interview_plan)
                    print("✅ Interview Conductor Agent initialized")
                except Exception as e:
                    print(f"⚠️  Interview Conductor Agent initialization failed: {e}")
                    self.interview_conductor = None
                
                # Update the recorder with the new interview components
                self.update_recorder_interview_components()
                
                return True
            else:
                # Handle different error field names - works for all document types
                error_msg = getattr(planning_result, 'error_message', None) or getattr(planning_result, 'error', 'Unknown error')
                print(f"❌ Interview plan creation failed: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Interview plan creation error: {e}")
            return False
    
    def _print_detailed_interview_plan(self):
        """Print the complete interview plan in a detailed, readable format."""

        # print(f"main.py: Class EnhancedInterviewPlatform._print_detailed_interview_plan called: Print the complete interview plan in a detailed, readable format.")
        try:
            if not self.interview_plan:
                print("❌ No interview plan available to display")
                return
            
            # Print interview overview
            print(f"\n🎯 INTERVIEW OVERVIEW")
            print(f"   Total Duration: {self.interview_plan.total_estimated_duration_minutes} minutes")
            print(f"   Total Sections: {len(self.interview_plan.interview_sections)}")
            
            # Print interview objectives
            if hasattr(self.interview_plan, 'interview_objectives') and self.interview_plan.interview_objectives:
                print(f"\n🎯 INTERVIEW OBJECTIVES:")
                for i, objective in enumerate(self.interview_plan.interview_objectives, 1):
                    print(f"   {i}. {objective}")
            
            # Print key focus areas
            if hasattr(self.interview_plan, 'key_focus_areas') and self.interview_plan.key_focus_areas:
                print(f"\n🔍 KEY FOCUS AREAS:")
                for i, area in enumerate(self.interview_plan.key_focus_areas, 1):
                    print(f"   {i}. {area}")
            
            # Print each section in detail
            print(f"\n📋 INTERVIEW SECTIONS:")
            for i, section in enumerate(self.interview_plan.interview_sections, 1):
                print(f"\n   {i}. {section.section_name.upper()}")
                print(f"      Phase: {section.phase.value}")
                print(f"      Duration: {section.estimated_duration_minutes} minutes")
                print(f"      Description: {section.description}")
                
                # Print section objectives
                if section.objectives:
                    print(f"      Objectives:")
                    for obj in section.objectives:
                        print(f"        • {obj}")
                
                # Print key evaluation points
                if section.key_evaluation_points:
                    print(f"      Key Evaluation Points:")
                    for point in section.key_evaluation_points:
                        print(f"        • {point}")
                
                # Print success criteria
                if section.success_criteria:
                    print(f"      Success Criteria:")
                    for criterion in section.success_criteria:
                        print(f"        • {criterion}")
                
                # Print questions in this section
                if section.questions:
                    print(f"      Questions ({len(section.questions)}):")
                    for j, question in enumerate(section.questions, 1):
                        print(f"\n        {j}. {question.question_text}")
                        print(f"           Type: {question.question_type.value}")
                        print(f"           Difficulty: {question.difficulty_level}/5")
                        print(f"           Estimated Time: {question.estimated_time_minutes} minutes")
                        
                        if question.skill_focus:
                            print(f"           Skill Focus: {', '.join(question.skill_focus)}")
                        
                        if question.evaluation_criteria:
                            print(f"           Evaluation Criteria: {', '.join(question.evaluation_criteria)}")
                else:
                    print(f"      Questions: None available")
            
            # Print evaluation priorities
            if hasattr(self.interview_plan, 'evaluation_priorities') and self.interview_plan.evaluation_priorities:
                print(f"\n📊 EVALUATION PRIORITIES:")
                for i, priority in enumerate(self.interview_plan.evaluation_priorities, 1):
                    print(f"   {i}. {priority}")
            
            # Print potential red flags
            if hasattr(self.interview_plan, 'potential_red_flags') and self.interview_plan.potential_red_flags:
                print(f"\n🚨 POTENTIAL RED FLAGS:")
                for i, flag in enumerate(self.interview_plan.potential_red_flags, 1):
                    print(f"   {i}. {flag}")
            
            # Print areas needing clarification
            if hasattr(self.interview_plan, 'clarification_needed') and self.interview_plan.clarification_needed:
                print(f"\n❓ AREAS NEEDING CLARIFICATION:")
                for i, area in enumerate(self.interview_plan.clarification_needed, 1):
                    print(f"   {i}. {area}")
            
            # Print interviewer notes
            if hasattr(self.interview_plan, 'interviewer_notes') and self.interview_plan.interviewer_notes:
                print(f"\n📝 INTERVIEWER NOTES:")
                for i, note in enumerate(self.interview_plan.interviewer_notes, 1):
                    print(f"   {i}. {note}")
            
            # Print recommended follow-up areas
            if hasattr(self.interview_plan, 'recommended_follow_up_areas') and self.interview_plan.recommended_follow_up_areas:
                print(f"\n🔄 RECOMMENDED FOLLOW-UP AREAS:")
                for i, area in enumerate(self.interview_plan.recommended_follow_up_areas, 1):
                    print(f"   {i}. {area}")
            
            print(f"\n✅ Interview plan display completed")
            
        except Exception as e:
            print(f"⚠️  Error displaying interview plan: {e}")
            import traceback
            traceback.print_exc()
    
    async def _auto_start_interview(self):
        """Automatically start interview using existing functions."""

        # print(f"main.py: Class EnhancedInterviewPlatform._auto_start_interview called: Automatically start interview using existing functions.")
        try:
            if not self.interview_plan:
                print("❌ Interview plan not available")
                return False
            
            if not self.interview_conductor:
                print("❌ Interview conductor not available")
                return False
            
            # Interview conductor is already initialized and ready
            self.interview_active = True
            print("✅ Interview ready to start")
            
            # Do not pre-ask the first plan question; let the conductor handle all questions
            try:
                opening_message = (
                    "Hello, I'm Sophia, your interview conductor today. "
                    "Before we begin, may I have your name? "
                )
                print(f"\n🤖 Interviewer: {opening_message}")
                if self.tts_service:
                    tts_result = self.tts_service.generate_speech(
                        opening_message,
                        play_immediately=True
                    )
                return True
            except Exception as e:
                print(f"⚠️  Could not generate opening message: {e}")
                return True
                
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
        # print(f"main.py: Class EnhancedAudioRecorder.__init__ called: Initialize the enhanced audio recorder.")
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
        # print(f"main.py: Class EnhancedAudioRecorder.check_microphone_access called: Test microphone access (same as original).")
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
        # print(f"main.py: Class EnhancedAudioRecorder.start_recording called: Start recording with enhanced logging.")
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
        # print(f"main.py: Class EnhancedAudioRecorder.stop_recording called: Stop recording and start transcription.")
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
        # print(f"main.py: Class EnhancedAudioRecorder._record_audio called: Record audio data (same as original implementation).")
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
        # print(f"main.py: Class EnhancedAudioRecorder._save_recording called: Save recorded audio frames to file.")
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
        # print(f"main.py: Class EnhancedAudioRecorder._start_transcription called: Start transcription in a separate thread.")
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
        # print(f"main.py: Class EnhancedAudioRecorder._transcribe_and_process called: Enhanced transcription with interview context processing.")
        try:
            if not self.current_filepath or not os.path.exists(self.current_filepath):
                print("⚠️  No valid audio file for transcription")
                return
            
            # Perform transcription (force English)
            result = self.stt_service.transcribe_audio(self.current_filepath, language='en')
            
            if result["success"]:
                transcription_text = result["text"]
                processing_time = result["processing_time"]
                
                if not self.minimal_logging:
                    print(f"✅ Transcription completed in {processing_time:.2f}s")
                
                print(f"🗣️  You: {transcription_text}")
                
                # Enhanced processing with interview conductor
                if self.interview_conductor and self.interview_plan:
                    # Process with interview conductor using synchronous approach
                    self._process_with_interview_conductor_sync(transcription_text)
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
        # print(f"main.py: Class EnhancedAudioRecorder._process_with_interview_conductor called: Process candidate response with interview conductor.")
        try:
            # Validate that conductor and plan are properly initialized
            if not self.interview_conductor or not self.interview_plan:
                print("⚠️ Interview conductor or plan not properly initialized")
                return
            
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
    
    def _process_with_interview_conductor_sync(self, transcription_text: str):
        """Synchronous version of interview conductor processing for use in threads."""
        # print(f"main.py: Class EnhancedAudioRecorder._process_with_interview_conductor_sync called: Synchronous version of interview conductor processing for use in threads.")
        try:
            # Validate that conductor and plan are properly initialized
            if not self.interview_conductor or not self.interview_plan:
                print("⚠️ Interview conductor or plan not properly initialized")
                return
            
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
            
            # Process with interview conductor using synchronous approach
            # We'll need to handle the async nature of the conductor differently
            try:
                # Create a new event loop for this thread if needed
                import asyncio
                loop_created = False
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop_created = True
                
                # Run the async conductor method in the event loop
                conductor_response = loop.run_until_complete(
                    self.interview_conductor.conduct_interview_turn(
                        transcription_text,
                        additional_context
                    )
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
                        # Handle completion synchronously
                        loop.run_until_complete(self._handle_interview_completion())
                        
                else:
                    print(f"❌ Interview conductor error: {conductor_response.error}")
                
                # Clean up the event loop if we created it
                if loop_created:
                    loop.close()
                    
            except Exception as e:
                print(f"❌ Interview conductor processing error: {e}")
                import traceback
                traceback.print_exc()
                
                # Clean up the event loop if we created it
                if loop_created:
                    loop.close()
                
        except Exception as e:
            print(f"❌ Interview conductor sync processing error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_interview_completion(self):
        """Handle interview completion and generate summary."""
        # print(f"main.py: Class EnhancedAudioRecorder._handle_interview_completion called: Handle interview completion and generate summary.")
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
        # print(f"main.py: Class EnhancedAudioRecorder._save_transcript called: Save transcription to file.")
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
    # print(f"main.py: Class main called: Entry point for the enhanced application.")
    try:
        # Create and run enhanced platform
        app = EnhancedInterviewPlatform(minimal_logging=False)
        asyncio.run(app.run())
    except Exception as e:
        print(f"❌ Enhanced Interview Platform failed to start - {e}")
        sys.exit(1)


if __name__ == "__main__":
    print(f"main.py: Class __name__ == __main__ called: Entry point for the enhanced application.")
    main()

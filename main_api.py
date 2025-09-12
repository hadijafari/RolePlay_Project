#!/usr/bin/env python3
"""
API-Compatible Interview Platform Main Module
Refactored version of main_test.py for FastAPI integration
"""

import asyncio
import os
import sys
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.deepgram_voice_agent_tab_service import DeepgramVoiceAgentTabService

# Import enhanced services for document analysis and interview planning
try:
    from agents.document_intelligence_agent import DocumentIntelligenceAgent
    from services.interview_planning_service import InterviewPlanningService
    from models.interview_models import ResumeAnalysis, JobDescriptionAnalysis, SimplifiedInterviewPlan, SimplifiedResumeAnalysis
    from agents.feedback_agent import FeedbackAgent
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    DocumentIntelligenceAgent = None
    InterviewPlanningService = None
    FeedbackAgent = None


class InterviewSession:
    """
    Manages a single interview session for API compatibility.
    Non-blocking version of the original interview functionality.
    """
    
    def __init__(self, session_id: str, enable_audio_recording: bool = True):
        self.session_id = session_id
        self.enable_audio_recording = enable_audio_recording
        self.status = "initialized"
        self.created_at = datetime.now()
        self.completed_at = None
        
        # Interview components
        self.resume_analysis = None
        self.job_analysis = None
        self.interview_plan = None
        self.questions = []
        self.feedback_agent = None
        self.conversation_logger = None
        self.voice_agent = None
        
        # Session data
        self.conversation_log = []
        self.audio_recording_path = None
        self.error_message = None
        
        print(f"📋 Interview session {session_id} initialized")
    
    async def setup_interview(self, rag_directory: Optional[str] = None) -> bool:
        """
        Set up the interview session with document processing and planning.
        Non-blocking version of the original setup.
        """
        try:
            self.status = "setting_up"
            
            # Process documents and generate plan
            self.resume_analysis, self.job_analysis, self.interview_plan = await self._process_documents_and_generate_plan(rag_directory)
            
            # Extract questions from interview plan
            if self.interview_plan:
                self.questions = self._extract_questions_from_interview_plan(self.interview_plan)
            
            # Initialize feedback agent and conversation logger
            await self._initialize_feedback_and_logging(rag_directory)
            
            self.status = "ready"
            print(f"✅ Interview session {self.session_id} setup completed")
            return True
            
        except Exception as e:
            self.status = "setup_failed"
            self.error_message = str(e)
            print(f"❌ Interview session {self.session_id} setup failed: {e}")
            return False
    
    async def start_interview(self) -> bool:
        """
        Start the interview session with voice agent.
        Returns immediately after connection, doesn't block.
        """
        try:
            if self.status != "ready":
                raise ValueError(f"Session not ready. Current status: {self.status}")
            
            self.status = "starting"
            
            # Initialize voice agent
            self.voice_agent = CustomDeepgramVoiceAgentTabService(
                questions=self.questions,
                feedback_agent=self.feedback_agent,
                conversation_logger=self.conversation_logger,
                enable_audio_recording=self.enable_audio_recording
            )
            
            # Connect to Deepgram (non-blocking)
            connected = await self.voice_agent.connect()
            if not connected:
                raise Exception("Failed to connect to Deepgram Voice Agent")
            
            self.status = "active"
            print(f"✅ Interview session {self.session_id} started successfully")
            return True
            
        except Exception as e:
            self.status = "start_failed"
            self.error_message = str(e)
            print(f"❌ Interview session {self.session_id} start failed: {e}")
            return False
    
    async def stop_interview(self, rag_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop the interview session and finalize recordings.
        """
        try:
            if self.voice_agent and self.voice_agent.is_connected:
                # Disconnect and export audio
                await self.voice_agent.disconnect(output_dir=rag_directory)
                
                # Get conversation history
                self.conversation_log = self.voice_agent.get_conversation_history()
            
            self.status = "completed"
            self.completed_at = datetime.now()
            
            # Determine audio recording path
            if rag_directory:
                rag_path = Path(rag_directory)
                audio_files = list(rag_path.glob("interview_recording_*.mp3"))
                if audio_files:
                    # Get the most recent audio file
                    self.audio_recording_path = str(max(audio_files, key=lambda x: x.stat().st_mtime))
            
            result = {
                "session_id": self.session_id,
                "status": self.status,
                "conversation_log": self.conversation_log,
                "audio_recording_path": self.audio_recording_path,
                "questions_count": len(self.questions),
                "duration_seconds": (self.completed_at - self.created_at).total_seconds() if self.completed_at else None
            }
            
            print(f"✅ Interview session {self.session_id} completed successfully")
            return result
            
        except Exception as e:
            self.status = "stop_failed"
            self.error_message = str(e)
            print(f"❌ Interview session {self.session_id} stop failed: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session status and information."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "questions_count": len(self.questions),
            "has_audio_recording": self.audio_recording_path is not None,
            "error_message": self.error_message,
            "enable_audio_recording": self.enable_audio_recording
        }
    
    async def _process_documents_and_generate_plan(self, rag_directory: Optional[str] = None) -> Tuple[Any, Any, Any]:
        """Process documents and generate interview plan (API-compatible version)."""
        if not ENHANCED_FEATURES_AVAILABLE:
            print("⚠️  Enhanced features not available - using hardcoded questions")
            return None, None, None
        
        try:
            # Initialize services
            document_agent = DocumentIntelligenceAgent()
            interview_planning_service = InterviewPlanningService()
            
            # Define RAG directory path
            if rag_directory:
                rag_dir = Path(rag_directory)
            else:
                rag_dir = Path(__file__).parent / "RAG"
            
            if not rag_dir.exists():
                print(f"❌ RAG directory not found: {rag_dir}")
                return None, None, None
            
            # Supported file extensions
            supported_extensions = ['.pdf', '.docx', '.txt']
            
            # Process resume
            resume_path = self._find_file_with_extensions(rag_dir, "resume", supported_extensions)
            if not resume_path:
                print(f"❌ Resume not found in {rag_dir}")
                return None, None, None
            
            resume_analysis = await self._process_resume(document_agent, str(resume_path))
            if not resume_analysis:
                return None, None, None
            
            # Process job description
            job_desc_path = self._find_file_with_extensions(rag_dir, "description", supported_extensions)
            if not job_desc_path:
                print(f"❌ Job description not found in {rag_dir}")
                return None, None, None
            
            job_analysis = await self._process_job_description(document_agent, str(job_desc_path))
            if not job_analysis:
                return None, None, None
            
            # Generate interview plan
            interview_plan = await self._generate_interview_plan(interview_planning_service, resume_analysis, job_analysis)
            
            return resume_analysis, job_analysis, interview_plan
            
        except Exception as e:
            print(f"❌ Error in document processing: {e}")
            return None, None, None
    
    def _find_file_with_extensions(self, directory: Path, base_name: str, extensions: list) -> Optional[Path]:
        """Find a file with the given base name and any of the supported extensions."""
        for ext in extensions:
            file_path = directory / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
        return None
    
    async def _process_resume(self, document_agent, file_path: str):
        """Process resume using document intelligence agent."""
        try:
            analysis_result = await document_agent.analyze_resume_simplified(file_path)
            
            if analysis_result.success:
                if hasattr(analysis_result, 'metadata') and 'resume_analysis' in analysis_result.metadata:
                    resume_data = analysis_result.metadata['resume_analysis']
                elif hasattr(analysis_result, 'data'):
                    resume_data = analysis_result.data
                else:
                    return None
                
                resume_analysis = SimplifiedResumeAnalysis(**resume_data)
                print(f"✅ Resume analysis completed for session {self.session_id}")
                return resume_analysis
            else:
                print(f"❌ Resume analysis failed for session {self.session_id}")
                return None
                
        except Exception as e:
            print(f"❌ Resume processing error for session {self.session_id}: {e}")
            return None
    
    async def _process_job_description(self, document_agent, file_path: str):
        """Process job description using document intelligence agent."""
        try:
            analysis_result = await document_agent.analyze_job_description(file_path)
            
            if analysis_result.success:
                if hasattr(analysis_result, 'metadata') and 'job_analysis' in analysis_result.metadata:
                    job_data = analysis_result.metadata['job_analysis']
                elif hasattr(analysis_result, 'data'):
                    job_data = analysis_result.data
                else:
                    return None
                
                job_analysis = JobDescriptionAnalysis(**job_data)
                print(f"✅ Job description analysis completed for session {self.session_id}")
                return job_analysis
            else:
                print(f"❌ Job description analysis failed for session {self.session_id}")
                return None
                
        except Exception as e:
            print(f"❌ Job description processing error for session {self.session_id}: {e}")
            return None
    
    async def _generate_interview_plan(self, interview_planning_service, resume_analysis, job_analysis):
        """Generate interview plan using interview planning service."""
        try:
            planning_result = await interview_planning_service.generate_simplified_interview_plan_from_simplified_resume(
                resume_analysis, job_analysis
            )
            
            if planning_result.success:
                if hasattr(planning_result, 'result_data'):
                    plan_data = planning_result.result_data
                elif hasattr(planning_result, 'data'):
                    plan_data = planning_result.data
                else:
                    return None
                
                interview_plan = SimplifiedInterviewPlan(**plan_data)
                print(f"✅ Interview plan created for session {self.session_id}")
                return interview_plan
            else:
                print(f"❌ Interview plan creation failed for session {self.session_id}")
                return None
                
        except Exception as e:
            print(f"❌ Interview plan creation error for session {self.session_id}: {e}")
            return None
    
    def _extract_questions_from_interview_plan(self, interview_plan):
        """Extract all questions from the interview plan."""
        if not interview_plan or not hasattr(interview_plan, 'interview_sections'):
            return []
        
        questions = []
        try:
            for section in interview_plan.interview_sections:
                if hasattr(section, 'questions') and section.questions:
                    for question in section.questions:
                        if hasattr(question, 'question_text') and question.question_text:
                            questions.append(question.question_text)
            
            print(f"📝 Extracted {len(questions)} questions for session {self.session_id}")
            return questions
            
        except Exception as e:
            print(f"❌ Error extracting questions for session {self.session_id}: {e}")
            return []
    
    async def _initialize_feedback_and_logging(self, rag_directory: Optional[str] = None):
        """Initialize feedback agent and conversation logger."""
        try:
            # Initialize feedback agent
            if FeedbackAgent:
                self.feedback_agent = FeedbackAgent()
                print(f"✅ Feedback agent initialized for session {self.session_id}")
            
            # Initialize conversation logger
            if rag_directory:
                rag_dir = Path(rag_directory)
            else:
                rag_dir = Path(__file__).parent / "RAG"
            
            rag_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = rag_dir / f"conversations_{self.session_id}_{timestamp}.txt"
            
            self.conversation_logger = ConversationLogger(str(log_file_path))
            print(f"✅ Conversation logger initialized for session {self.session_id}")
            
        except Exception as e:
            print(f"❌ Error initializing feedback/logging for session {self.session_id}: {e}")


class ConversationLogger:
    """Handles logging of conversations to file (API-compatible version)."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.question_counter = 0
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure the log file exists and is ready for writing."""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Interview Conversation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"❌ Failed to initialize conversation log: {e}")
    
    def log_qa_feedback(self, question: str, answer: str, feedback: str):
        """Log a Q&A pair with feedback."""
        try:
            self.question_counter += 1
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"Question {self.question_counter}: {question}\n")
                f.write(f"Answer {self.question_counter}: {answer}\n")
                f.write(f"Feedback {self.question_counter}: {feedback}\n")
                f.write("-" * 80 + "\n\n")
            
        except Exception as e:
            print(f"❌ Failed to log Q&A pair: {e}")


class CustomDeepgramVoiceAgentTabService(DeepgramVoiceAgentTabService):
    """Custom voice agent for API compatibility (simplified version)."""
    
    def __init__(self, questions=None, feedback_agent=None, conversation_logger=None, enable_audio_recording=True):
        super().__init__(enable_audio_recording=enable_audio_recording)
        self.dynamic_questions = questions or []
        self.feedback_agent = feedback_agent
        self.conversation_logger = conversation_logger
        
        # Role switch detection variables (simplified for API use)
        self.current_role = None
        self.assistant_message_buffer = ""
        self.user_message_buffer = ""
        self.complete_question = None
        self.complete_answer = None
        self.question_counter = 0
    
    def _get_hardcoded_system_prompt(self) -> str:
        """Return the system prompt with dynamic questions from interview plan."""
        if not self.dynamic_questions:
            return super()._get_hardcoded_system_prompt()
        
        questions_text = "\n".join([f"- {q}" for q in self.dynamic_questions])
        
        return f"""You are conducting a natural interview conversation. You need to ask these questions in this exact order, but make it sound like a normal conversation:

{questions_text}

CRITICAL INSTRUCTIONS:
- The agent will say the initial message and asks that the interviewee introduces himself.
- After the interviewee introduces himself, the agent will respond in one or 2 sentence and then asks the first question.
- Ask these questions in the exact order listed above
- Make it sound like a natural conversation - don't say question numbers or "first question", "second question", etc.
- After each answer, provide brief constructive feedback (1-2 sentences) when the answer could be improved or when clarification would be helpful
- If the answer is good and complete, simply acknowledge it briefly and move to the next question without excessive praise
- Be encouraging and supportive, but focus on helping the interviewee improve rather than just praising
- Do NOT ask follow-up questions or elaborations
- Do NOT skip questions or ask them out of order
- Do NOT create your own questions
- Keep the conversation flowing naturally
- Start with: "{self.dynamic_questions[0] if self.dynamic_questions else 'Tell me about your experience and background.'}"

You are a friendly, professional interviewer conducting a comprehensive interview to assess the candidate's technical skills, experience, and cultural fit for the role."""


# Global session manager for API compatibility
active_sessions: Dict[str, InterviewSession] = {}


async def create_interview_session(session_id: str, enable_audio_recording: bool = True) -> InterviewSession:
    """Create a new interview session for API use."""
    session = InterviewSession(session_id, enable_audio_recording)
    active_sessions[session_id] = session
    return session


async def get_interview_session(session_id: str) -> Optional[InterviewSession]:
    """Get an existing interview session."""
    return active_sessions.get(session_id)


async def cleanup_session(session_id: str) -> bool:
    """Clean up and remove a session."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if session.voice_agent and session.voice_agent.is_connected:
            await session.voice_agent.disconnect()
        del active_sessions[session_id]
        return True
    return False


# Example usage function for testing
async def test_api_compatibility():
    """Test function to verify API compatibility."""
    print("🧪 Testing API compatibility...")
    
    # Create a session
    session = await create_interview_session("test-session-001")
    
    # Setup interview
    setup_success = await session.setup_interview()
    print(f"Setup success: {setup_success}")
    
    # Get status
    status = session.get_status()
    print(f"Session status: {status}")
    
    # Cleanup
    await cleanup_session("test-session-001")
    print("✅ API compatibility test completed")


if __name__ == "__main__":
    # This allows the module to be tested independently
    asyncio.run(test_api_compatibility())

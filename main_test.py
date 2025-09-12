#!/usr/bin/env python3
"""
Enhanced test file for Deepgram Voice Agent with dynamic interview plan generation.
"""

import asyncio
import os
import sys
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

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
    print("✅ Enhanced interview features loaded successfully")
except ImportError as e:
    print(f"⚠️  Enhanced features not available: {e}")
    print("⚠️  Running in basic mode with hardcoded questions")
    ENHANCED_FEATURES_AVAILABLE = False
    DocumentIntelligenceAgent = None
    InterviewPlanningService = None
    FeedbackAgent = None


async def process_documents_and_generate_plan():
    """Process resume and job description, then generate interview plan."""
    if not ENHANCED_FEATURES_AVAILABLE:
        print("⚠️  Enhanced features not available - using hardcoded questions")
        return None, None, None
    
    try:
        # Initialize services
        print("🔧 Initializing document analysis services...")
        document_agent = DocumentIntelligenceAgent()
        interview_planning_service = InterviewPlanningService()
        
        # Define RAG directory path
        rag_dir = Path(__file__).parent / "RAG"
        
        if not rag_dir.exists():
            print(f"❌ RAG directory not found: {rag_dir}")
            return None, None, None
        
        print(f"📁 Loading documents from: {rag_dir}")
        
        # Supported file extensions
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        # Step 1: Process resume
        resume_path = find_file_with_extensions(rag_dir, "resume", supported_extensions)
        if not resume_path:
            print(f"❌ Resume not found. Looking for: resume.pdf, resume.docx, or resume.txt in {rag_dir}")
            return None, None, None
        
        print(f"📄 Processing resume: {resume_path}")
        resume_analysis = await process_resume(document_agent, str(resume_path))
        if not resume_analysis:
            return None, None, None
        
        # Step 2: Process job description
        job_desc_path = find_file_with_extensions(rag_dir, "description", supported_extensions)
        if not job_desc_path:
            print(f"❌ Job description not found. Looking for: description.pdf, description.docx, or description.txt in {rag_dir}")
            return None, None, None
        
        print(f"📋 Processing job description: {job_desc_path}")
        job_analysis = await process_job_description(document_agent, str(job_desc_path))
        if not job_analysis:
            return None, None, None
        
        # Step 3: Generate interview plan
        print("🎯 Creating interview plan...")
        interview_plan = await generate_interview_plan(interview_planning_service, resume_analysis, job_analysis)
        if not interview_plan:
            return None, None, None
        
        print("✅ Document processing and interview plan generation completed!")
        return resume_analysis, job_analysis, interview_plan
        
    except Exception as e:
        print(f"❌ Error in document processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def find_file_with_extensions(directory: Path, base_name: str, extensions: list) -> Path:
    """Find a file with the given base name and any of the supported extensions."""
    for ext in extensions:
        file_path = directory / f"{base_name}{ext}"
        if file_path.exists():
            return file_path
    return None


async def process_resume(document_agent, file_path: str):
    """Process resume using simplified document intelligence agent."""
    try:
        print("⏳ Analyzing resume (simplified)...")
        
        # Use simplified document analysis logic
        analysis_result = await document_agent.analyze_resume_simplified(file_path)
        
        if analysis_result.success:
            # Extract resume analysis from metadata
            if hasattr(analysis_result, 'metadata') and 'resume_analysis' in analysis_result.metadata:
                resume_data = analysis_result.metadata['resume_analysis']
            elif hasattr(analysis_result, 'data'):
                resume_data = analysis_result.data
            else:
                print("⚠️  Resume analysis data not found in expected location")
                return None
            
            # Create SimplifiedResumeAnalysis object from the data
            try:
                resume_analysis = SimplifiedResumeAnalysis(**resume_data)
                print(f"✅ Simplified resume analysis completed")
                print(f"   📊 Skills identified: {len(resume_analysis.technical_skills)}")
                
                # Save simplified resume analysis to file
                with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\simplified_resume_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(resume_analysis.model_dump(mode='json'), indent=2))
                print(f"In process_resume method of main_test.py: Simplified resume analysis saved to file")
                
                # Safely access experience years
                try:
                    experience_years = resume_analysis.total_experience_years
                    print(f"   💼 Experience years: {experience_years}")
                except AttributeError:
                    print(f"   💼 Experience years: Not available")
                
                return resume_analysis
            except Exception as e:
                print(f"❌ Failed to create SimplifiedResumeAnalysis object: {e}")
                return None
        else:
            error_msg = getattr(analysis_result, 'error_message', None) or getattr(analysis_result, 'error', 'Unknown error')
            print(f"❌ Resume analysis failed: {error_msg}")
            return None
            
    except Exception as e:
        print(f"❌ Resume processing error: {e}")
        return None


async def process_job_description(document_agent, file_path: str):
    """Process job description using document intelligence agent."""
    try:
        print("⏳ Analyzing job description...")
        
        # Use document analysis logic
        analysis_result = await document_agent.analyze_job_description(file_path)
        
        if analysis_result.success:
            # Extract job description analysis from metadata
            if hasattr(analysis_result, 'metadata') and 'job_analysis' in analysis_result.metadata:
                job_data = analysis_result.metadata['job_analysis']
            elif hasattr(analysis_result, 'data'):
                job_data = analysis_result.data
            else:
                print("⚠️  Job description analysis data not found in expected location")
                return None
            
            # Create JobDescriptionAnalysis object from the data
            try:
                job_analysis = JobDescriptionAnalysis(**job_data)
                print(f"✅ Job description analysis completed")
                
                # Save job description analysis to file
                with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\job_description_analysis_{job_analysis.analysis_timestamp.strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(job_analysis.model_dump(mode='json'), indent=2))
                print(f"In process_job_description method of main_test.py: Job description analysis saved to file")
                
                # Safely access job title
                try:
                    job_title = job_analysis.job_title
                    print(f"   🎯 Role: {job_title}")
                except AttributeError:
                    print(f"   🎯 Role: Not specified")
                
                return job_analysis
            except Exception as e:
                print(f"❌ Failed to create JobDescriptionAnalysis object: {e}")
                return None
        else:
            error_msg = getattr(analysis_result, 'error_message', None) or getattr(analysis_result, 'error', 'Unknown error')
            print(f"❌ Job description analysis failed: {error_msg}")
            return None
            
    except Exception as e:
        print(f"❌ Job description processing error: {e}")
        return None


async def generate_interview_plan(interview_planning_service, resume_analysis, job_analysis):
    """Generate simplified interview plan using interview planning service."""
    try:
        # Use simplified interview planning logic with simplified resume analysis
        planning_result = await interview_planning_service.generate_simplified_interview_plan_from_simplified_resume(
            resume_analysis,
            job_analysis
        )
        
        if planning_result.success:
            # Extract interview plan from result_data
            if hasattr(planning_result, 'result_data'):
                plan_data = planning_result.result_data
            elif hasattr(planning_result, 'data'):
                plan_data = planning_result.data
            else:
                print("⚠️  Interview plan data not found in expected location")
                return None
            
            # Create SimplifiedInterviewPlan object from the data
            try:
                interview_plan = SimplifiedInterviewPlan(**plan_data)
                print(f"✅ Simplified interview plan created")
                
                # Save interview plan to file
                with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\simplified_interview_plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(interview_plan.model_dump(mode='json'), indent=2, default=str))                          
                print(f"In generate_interview_plan method of main_test.py: Simplified interview plan saved to file")
                
                # Safely access sections
                try:
                    sections_count = len(interview_plan.interview_sections)
                    print(f"   📋 Sections: {sections_count}")
                except AttributeError:
                    print(f"   📋 Sections: Not available")
                
                return interview_plan
            except Exception as e:
                print(f"❌ Failed to create SimplifiedInterviewPlan object: {e}")
                return None
        else:
            error_msg = getattr(planning_result, 'error_message', None) or getattr(planning_result, 'error', 'Unknown error')
            print(f"❌ Interview plan creation failed: {error_msg}")
            return None
            
    except Exception as e:
        print(f"❌ Interview plan creation error: {e}")
        return None


def extract_questions_from_interview_plan(interview_plan):
    """Extract all questions from the simplified interview plan."""
    if not interview_plan or not hasattr(interview_plan, 'interview_sections'):
        print("❌ No interview plan or sections available")
        return []
    
    questions = []
    try:
        for section in interview_plan.interview_sections:
            if hasattr(section, 'questions') and section.questions:
                for question in section.questions:
                    if hasattr(question, 'question_text') and question.question_text:
                        questions.append(question.question_text)
        
        print(f"📝 Extracted {len(questions)} questions from simplified interview plan")
        
        # Print all questions for verification
        print("\n" + "="*80)
        print("📋 EXTRACTED QUESTIONS FROM SIMPLIFIED INTERVIEW PLAN:")
        print("="*80)
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")
        print("="*80)
        
        return questions
        
    except Exception as e:
        print(f"❌ Error extracting questions: {e}")
        return []


class MockConversationText:
    """Mock conversation text object for testing."""
    def __init__(self, role, content):
        self.role = role
        self.content = content


class ConversationLogger:
    """Handles logging of conversations to file."""
    
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
            print(f"📝 Conversation log initialized: {self.log_file_path}")
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
            
            print(f"📝 Logged Q&A pair {self.question_counter} to conversation file")
            
        except Exception as e:
            print(f"❌ Failed to log Q&A pair: {e}")


class CustomDeepgramVoiceAgentTabService(DeepgramVoiceAgentTabService):
    """Custom voice agent that uses dynamic questions from interview plan and provides feedback."""
    
    def __init__(self, questions=None, feedback_agent=None, conversation_logger=None):
        super().__init__()
        self.dynamic_questions = questions or []
        self.feedback_agent = feedback_agent
        self.conversation_logger = conversation_logger
        
        # Role switch detection variables
        self.current_role = None
        self.assistant_message_buffer = ""  # Accumulates interviewer message chunks
        self.user_message_buffer = ""       # Accumulates interviewee message chunks
        self.complete_question = None       # Last complete question
        self.complete_answer = None         # Last complete answer
        self.question_counter = 0
    
    def _get_hardcoded_system_prompt(self) -> str:
        """Return the system prompt with dynamic questions from interview plan."""
        if not self.dynamic_questions:
            # Fallback to original hardcoded questions if no dynamic questions
            return super()._get_hardcoded_system_prompt()
        
        # Build questions list for the prompt
        questions_list = []
        for i, question in enumerate(self.dynamic_questions, 1):
            questions_list.append(f"- {question}")
        
        questions_text = "\n".join(questions_list)
        
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
    
    async def _process_qa_pair(self, question: str, answer: str):
        """Process a Q&A pair and generate feedback asynchronously."""
        if not self.feedback_agent or not self.conversation_logger:
            return
        
        try:
            print(f"🔄 Processing Q&A pair {self.question_counter + 1}...")
            
            # Generate feedback asynchronously
            feedback_data = await self.feedback_agent.generate_feedback(
                question=question,
                answer=answer,
                question_number=self.question_counter + 1
            )
            
            # Format feedback for display
            formatted_feedback = self.feedback_agent.format_feedback_for_display(feedback_data)
            
            # Log to conversation file
            self.conversation_logger.log_qa_feedback(question, answer, formatted_feedback)
            
            # Display feedback summary
            print(f"📊 Feedback generated for Q&A pair {self.question_counter + 1}")
            print(f"   Overall Score: {feedback_data.get('overall_score', 0.5):.2f}/1.0")
            print(f"   Summary: {feedback_data.get('summary', 'No summary available')}")
            
        except Exception as e:
            print(f"❌ Error processing Q&A pair: {e}")
    
    def _register_event_handlers(self):
        """Register all event handlers with custom Q&A tracking."""
        
        # Audio data handler - receives audio from agent
        def on_audio_data(_, data, **kwargs):
            """Handle incoming audio data from agent."""
            self.audio_buffer.extend(data)
            
            # Queue audio for playback
            if len(self.audio_buffer) >= self.chunk_size * 2:
                chunk = bytes(self.audio_buffer[:self.chunk_size * 2])
                self.audio_output_queue.put(chunk)
                self.audio_buffer = self.audio_buffer[self.chunk_size * 2:]
        
        # Enhanced conversation text handler with role switch detection
        def on_conversation_text(_, conversation_text, **kwargs):
            """Log conversation text and track Q&A pairs using role switch detection."""
            text_data = {
                "role": conversation_text.role,
                "content": conversation_text.content,
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_log.append(text_data)
            
            # Role switch detection logic
            incoming_role = conversation_text.role
            
            # Check if role has switched
            if self.current_role is not None and self.current_role != incoming_role:
                print(f"🔄 Role switch detected: {self.current_role} → {incoming_role}")
                
                # Process complete messages when role switches
                if self.current_role == "assistant" and incoming_role == "user":
                    # Interviewer finished speaking, user starting to speak
                    self.complete_question = self.assistant_message_buffer.strip()
                    self.assistant_message_buffer = ""  # Reset buffer
                    print(f"📝 Complete question captured: {self.complete_question[:400]}...")
                    
                elif self.current_role == "user" and incoming_role == "assistant":
                    # User finished speaking, interviewer starting to speak
                    self.complete_answer = self.user_message_buffer.strip()
                    self.user_message_buffer = ""  # Reset buffer
                    print(f"📝 Complete answer captured: {self.complete_answer[:400]}...")
                    
                    # Process Q&A pair if we have both complete question and answer
                    if self.complete_question and self.complete_answer:
                        print(f"🔄 Processing complete Q&A pair {self.question_counter + 1}...")
                        
                        # Process Q&A pair asynchronously (non-blocking)
                        try:
                            # Try to get the running event loop
                            loop = asyncio.get_running_loop()
                            # Schedule the coroutine to run in the event loop
                            asyncio.run_coroutine_threadsafe(
                                self._process_qa_pair(self.complete_question, self.complete_answer), 
                                loop
                            )
                        except RuntimeError:
                            # No running loop, create a new task in a thread
                            def run_feedback():
                                asyncio.run(self._process_qa_pair(self.complete_question, self.complete_answer))
                            threading.Thread(target=run_feedback, daemon=True).start()
                        
                        # Reset for next Q&A pair
                        self.complete_question = None
                        self.complete_answer = None
                        self.question_counter += 1
            
            # Update current role
            self.current_role = incoming_role
            
            # Accumulate message chunks in appropriate buffer
            if incoming_role == "assistant":
                # Add space between chunks if buffer is not empty
                if self.assistant_message_buffer:
                    self.assistant_message_buffer += " "
                self.assistant_message_buffer += conversation_text.content
                print(f"🤖 Interviewer: {conversation_text.content}")
                
                if self.waiting_for_response:
                    self.waiting_for_response = False
                    self.logger.debug("Turn completed by ConversationText (assistant)")
                    
                    # Save state after successful response
                    self._save_conversation_state()
                    
                    # Reset retry count on success
                    self.retry_count = 0
                    
            elif incoming_role == "user":
                # Add space between chunks if buffer is not empty
                if self.user_message_buffer:
                    self.user_message_buffer += " "
                self.user_message_buffer += conversation_text.content
                print(f"🗣️  You: {conversation_text.content}")
        
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
            
            # Save state after successful turn completion
            self._save_conversation_state()
            
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
            """Handle errors with retry logic."""
            error_msg = str(error)
            self.logger.error(f"Voice Agent error: {error}")
            print(f"❌ Error: {error}")
            
            # Check if it's a think provider error (503 Service Unavailable)
            if "THINK_REQUEST_FAILED" in error_msg or "503" in error_msg or "Service Unavailable" in error_msg:
                print("🔄 Think provider error detected - attempting recovery...")
                
                # Handle think provider failure with retry logic
                if self._handle_think_provider_failure(error_msg):
                    # Retry the current request
                    print("🔄 Retrying request...")
                    self.waiting_for_response = False
                    return
                else:
                    # Use fallback response
                    fallback_response = self._get_fallback_response()
                    print(f"🤖 Fallback: {fallback_response}")
                    
                    # Add fallback to conversation log
                    self.conversation_log.append({
                        "role": "assistant",
                        "content": fallback_response,
                        "timestamp": datetime.now().isoformat(),
                        "is_fallback": True
                    })
                    
                    # Reset retry count for next attempt
                    self.retry_count = 0
                    self.stats["recovery_count"] += 1
            else:
                # For other errors, just log and continue
                print("⚠️ Non-critical error - continuing...")
            
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
        
        # Import AgentWebSocketEvents for event registration
        from deepgram import AgentWebSocketEvents
        
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
        print("✅ Event handlers registered with Q&A tracking")
        
        self.logger.debug("Event handlers registered with Q&A tracking")


async def main():
    """Main test function with dynamic interview plan generation."""
    print("🧪 ENHANCED DEEPGRAM VOICE AGENT TEST")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Process documents and generate interview plan
    print("🚀 STEP 1: Processing documents and generating interview plan...")
    resume_analysis, job_analysis, interview_plan = await process_documents_and_generate_plan()
    
    # Step 2: Extract questions from interview plan
    questions = []
    if interview_plan:
        questions = extract_questions_from_interview_plan(interview_plan)
    
    if not questions:
        print("⚠️  No questions extracted - falling back to hardcoded questions")
        print("📝 Using original 25 hardcoded questions")
    else:
        print(f"✅ Successfully extracted {len(questions)} questions from interview plan")
    
    # Step 3: Initialize feedback agent and conversation logger
    print("\n🔧 STEP 2: Initializing feedback agent and conversation logger...")
    feedback_agent = None
    conversation_logger = None
    
    if FeedbackAgent:
        try:
            feedback_agent = FeedbackAgent()
            print("✅ Feedback agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize feedback agent: {e}")
            print("⚠️  Continuing without feedback functionality")
    else:
        print("⚠️  FeedbackAgent not available - running without feedback")
    
    # Create conversation log file in RAG directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_dir = Path(__file__).parent / "RAG"
    rag_dir.mkdir(exist_ok=True)  # Ensure RAG directory exists
    log_file_path = rag_dir / f"conversations_{timestamp}.txt"
    conversation_logger = ConversationLogger(str(log_file_path))
    
    # Step 4: Initialize custom voice agent with dynamic questions and feedback
    print("\n🔧 STEP 3: Initializing voice agent with dynamic questions and feedback...")
    voice_agent = CustomDeepgramVoiceAgentTabService(
        questions=questions,
        feedback_agent=feedback_agent,
        conversation_logger=conversation_logger
    )
    
    try:
        # Step 5: Connect to Deepgram Voice Agent and start interview
        print("\n🔧 STEP 4: Connecting to Deepgram Voice Agent...")
        if await voice_agent.connect():
            print("✅ Connected to Deepgram Voice Agent")
            print("\n" + "="*60)
            print("🎤 INTERVIEW READY")
            print("📝 Hold TAB to speak, release to send")
            print("📊 Feedback will be generated for each Q&A pair")
            print(f"📁 Conversation log: {log_file_path}")
            print("❌ Press Ctrl+C to exit")
            print("="*60)
            
            # Start the interview loop
            await voice_agent.start_interview()
            
        else:
            print("❌ Failed to connect to Deepgram Voice Agent")
            
    except KeyboardInterrupt:
        print("\n🛑 Interview stopped by user")
        print(f"📁 Conversation log saved: {log_file_path}")
    except Exception as e:
        print(f"❌ Error during interview: {e}")
        print(f"📁 Conversation log saved: {log_file_path}")
    finally:
        await voice_agent.disconnect()
        print(f"📁 Final conversation log: {log_file_path}")


if __name__ == "__main__":
    asyncio.run(main())

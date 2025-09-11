#!/usr/bin/env python3
"""
Enhanced test file for Deepgram Voice Agent with dynamic interview plan generation.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.deepgram_voice_agent_tab_service import DeepgramVoiceAgentTabService

# Import enhanced services for document analysis and interview planning
try:
    from agents.document_intelligence_agent import DocumentIntelligenceAgent
    from services.interview_planning_service import InterviewPlanningService
    from models.interview_models import ResumeAnalysis, JobDescriptionAnalysis
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced interview features loaded successfully")
except ImportError as e:
    print(f"⚠️  Enhanced features not available: {e}")
    print("⚠️  Running in basic mode with hardcoded questions")
    ENHANCED_FEATURES_AVAILABLE = False
    DocumentIntelligenceAgent = None
    InterviewPlanningService = None


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
    """Process resume using document intelligence agent."""
    try:
        print("⏳ Analyzing resume...")
        
        # Use document analysis logic
        analysis_result = await document_agent.analyze_resume(file_path)
        
        if analysis_result.success:
            # Extract resume analysis from metadata
            if hasattr(analysis_result, 'metadata') and 'resume_analysis' in analysis_result.metadata:
                resume_data = analysis_result.metadata['resume_analysis']
            elif hasattr(analysis_result, 'data'):
                resume_data = analysis_result.data
            else:
                print("⚠️  Resume analysis data not found in expected location")
                return None
            
            # Create ResumeAnalysis object from the data
            try:
                resume_analysis = ResumeAnalysis(**resume_data)
                print(f"✅ Resume analysis completed")
                print(f"   📊 Skills identified: {len(resume_analysis.technical_skills)}")
                
                # Save resume analysis to file
                with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\resume_analysis_{resume_analysis.analysis_timestamp.strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(resume_analysis.model_dump(mode='json'), indent=2))
                print(f"In process_resume method of main_test.py: Resume analysis saved to file")
                
                # Safely access experience years
                try:
                    if resume_analysis.career_progression:
                        experience_years = resume_analysis.career_progression.total_experience_years
                        print(f"   💼 Experience years: {experience_years}")
                    else:
                        print(f"   💼 Experience years: Not specified")
                except AttributeError:
                    print(f"   💼 Experience years: Not available")
                
                return resume_analysis
            except Exception as e:
                print(f"❌ Failed to create ResumeAnalysis object: {e}")
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
    """Generate interview plan using interview planning service."""
    try:
        # Use existing interview planning logic
        planning_result = await interview_planning_service.generate_interview_plan(
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
            
            # Create InterviewPlan object from the data
            try:
                from models.interview_models import InterviewPlan
                interview_plan = InterviewPlan(**plan_data)
                print(f"✅ Interview plan created")
                
                # Save interview plan to file
                with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(interview_plan.model_dump(mode='json'), indent=2, default=str))                          
                print(f"In generate_interview_plan method of main_test.py: Interview plan saved to file")
                
                # Safely access sections
                try:
                    sections_count = len(interview_plan.interview_sections)
                    print(f"   📋 Sections: {sections_count}")
                except AttributeError:
                    print(f"   📋 Sections: Not available")
                
                return interview_plan
            except Exception as e:
                print(f"❌ Failed to create InterviewPlan object: {e}")
                return None
        else:
            error_msg = getattr(planning_result, 'error_message', None) or getattr(planning_result, 'error', 'Unknown error')
            print(f"❌ Interview plan creation failed: {error_msg}")
            return None
            
    except Exception as e:
        print(f"❌ Interview plan creation error: {e}")
        return None


def extract_questions_from_interview_plan(interview_plan):
    """Extract all questions from the interview plan."""
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
        
        print(f"📝 Extracted {len(questions)} questions from interview plan")
        
        # Print all questions for verification
        print("\n" + "="*80)
        print("📋 EXTRACTED QUESTIONS FROM INTERVIEW PLAN:")
        print("="*80)
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")
        print("="*80)
        
        return questions
        
    except Exception as e:
        print(f"❌ Error extracting questions: {e}")
        return []


class CustomDeepgramVoiceAgentTabService(DeepgramVoiceAgentTabService):
    """Custom voice agent that uses dynamic questions from interview plan."""
    
    def __init__(self, questions=None):
        super().__init__()
        self.dynamic_questions = questions or []
    
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
        
        return f"""You are conducting a natural Electronic Engineering interview conversation. You need to ask these questions in this exact order, but make it sound like a normal conversation:

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
- Start with: "{self.dynamic_questions[0] if self.dynamic_questions else 'Tell me about your experience with microcontroller programming and which platforms you\'ve worked with.'}"

You are a friendly, professional interviewer conducting an Electronic Engineering interview focused on Microcontroller Programming, PCB Design, and Embedded Systems."""


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
    
    # Step 3: Initialize custom voice agent with dynamic questions
    print("\n🔧 STEP 2: Initializing voice agent with dynamic questions...")
    voice_agent = CustomDeepgramVoiceAgentTabService(questions)
    
    try:
        # Connect to Deepgram Voice Agent
        if await voice_agent.connect():
            print("✅ Connected to Deepgram Voice Agent")
            print("\n" + "="*60)
            print("🎤 INTERVIEW READY")
            print("📝 Hold TAB to speak, release to send")
            print("❌ Press Ctrl+C to exit")
            print("="*60)
            
            # Start the interview loop
            await voice_agent.start_interview()
            
        else:
            print("❌ Failed to connect to Deepgram Voice Agent")
            
    except KeyboardInterrupt:
        print("\n🛑 Interview stopped by user")
    except Exception as e:
        print(f"❌ Error during interview: {e}")
    finally:
        await voice_agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

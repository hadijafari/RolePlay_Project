"""
Interview Planning Service
Generates comprehensive interview plans based on resume and job description analysis.
Creates structured interview sections, questions, and evaluation criteria.
"""

import os
import sys
import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Interview Planning Service: Missing dependencies: {e}")
    sys.exit(1)

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from models.interview_models import (
        ResumeAnalysis, JobDescriptionAnalysis, CandidateMatch,
        InterviewPlan, InterviewSection, Question, InterviewPhase, QuestionType,
        ProcessingResult
    )
except ImportError as e:
    print(f"Interview Planning Service: Missing models: {e}")
    sys.exit(1)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class InterviewPlanningConfig:
    """Configuration for interview planning service."""
    
    # OpenAI settings
    MODEL = "gpt-4o-mini"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.4  # Balanced creativity for question generation
    
    # Interview structure settings
    # DEFAULT_INTERVIEW_DURATION = 60  # minutes
    # OPENING_DURATION = 5    # minutes
    # TECHNICAL_DURATION = 20  # minutes  
    # EXPERIENCE_DURATION = 15 # minutes
    # BEHAVIORAL_DURATION = 15 # minutes
    # CLOSING_DURATION = 5    # minutes
    
   # set it to 30 minutes in total
    DEFAULT_INTERVIEW_DURATION = 30  # minutes
    OPENING_DURATION = 3    # minutes
    TECHNICAL_DURATION = 8  # minutes  
    EXPERIENCE_DURATION = 8 # minutes
    BEHAVIORAL_DURATION = 8 # minutes
    CLOSING_DURATION = 3    # minutes


    # Question generation settings
    QUESTIONS_PER_SECTION = 5
    BACKUP_QUESTIONS_PER_SECTION = 3
    MIN_QUESTION_TIME = 2   # minutes
    MAX_QUESTION_TIME = 10  # minutes
    
    # Processing settings
    MAX_RETRY_ATTEMPTS = 3
    TIMEOUT = 90.0  # seconds


class InterviewPlanningService:
    """
    Service for generating comprehensive interview plans based on document analysis.
    
    This service:
    1. Takes resume and job description analysis as input
    2. Generates structured interview sections
    3. Creates relevant questions for each section
    4. Provides evaluation criteria and success metrics
    5. Customizes interview approach based on candidate-job match
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the interview planning service."""
        print("interview_planning_service.py: Class InterviewPlanningService.__init__ called: Initialize the interview planning service.")
        self.config = InterviewPlanningConfig()
        self.logger = self._setup_logger()
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Interview Planning Service: OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Question templates and frameworks
        self.question_frameworks = self._initialize_question_frameworks()
        
        # Statistics tracking
        self.stats = {
            "plans_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_processing_time": 0.0,
            "average_questions_per_plan": 0.0,
            "custom_questions_generated": 0
        }
        
        self.logger.info("Interview Planning Service: Initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the service."""
        logger = logging.getLogger("InterviewPlanningService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Interview Planning: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_question_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize question frameworks for different interview phases."""
        
        return {
            "opening": {
                "objectives": [
                    "Make candidate comfortable",
                    "Understand motivation for role",
                    "Get overview of background",
                    "Set interview tone"
                ],
                "question_types": [QuestionType.OPEN_ENDED],
                "sample_questions": [
                    "Tell me about yourself and what interests you about this role.",
                    "What do you know about our company and why do you want to work here?",
                    "Walk me through your career journey and what led you to apply for this position."
                ]
            },
            "technical": {
                "objectives": [
                    "Assess technical competency",
                    "Evaluate problem-solving skills",
                    "Test depth of knowledge",
                    "Understand technical experience"
                ],
                "question_types": [QuestionType.TECHNICAL, QuestionType.SITUATIONAL],
                "sample_questions": [
                    "Describe a challenging technical problem you've solved recently.",
                    "How would you approach designing a system to handle [specific requirement]?",
                    "Walk me through your experience with [specific technology]."
                ]
            },
            "experience": {
                "objectives": [
                    "Validate resume claims",
                    "Understand work approach",
                    "Assess leadership potential",
                    "Evaluate cultural fit"
                ],
                "question_types": [QuestionType.BEHAVIORAL, QuestionType.SITUATIONAL],
                "sample_questions": [
                    "Tell me about your most significant professional achievement.",
                    "Describe a time when you had to work with a difficult team member.",
                    "How do you handle competing priorities and tight deadlines?"
                ]
            },
            "behavioral": {
                "objectives": [
                    "Assess soft skills",
                    "Evaluate cultural fit",
                    "Understand work style",
                    "Test emotional intelligence"
                ],
                "question_types": [QuestionType.BEHAVIORAL],
                "sample_questions": [
                    "Tell me about a time when you failed at something. How did you handle it?",
                    "Describe a situation where you had to adapt to significant change.",
                    "How do you approach giving feedback to colleagues?"
                ]
            },
            "closing": {
                "objectives": [
                    "Answer candidate questions",
                    "Gauge continued interest",
                    "Explain next steps",
                    "Leave positive impression"
                ],
                "question_types": [QuestionType.OPEN_ENDED],
                "sample_questions": [
                    "What questions do you have about the role or our company?",
                    "Is there anything we haven't covered that you'd like to discuss?",
                    "What are your salary expectations for this position?"
                ]
            }
        }
    
    async def generate_interview_plan(self,
                                    resume_analysis: ResumeAnalysis,
                                    job_analysis: JobDescriptionAnalysis,
                                    candidate_match: Optional[CandidateMatch] = None,
                                    interview_duration: int = None,
                                    custom_requirements: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Generate comprehensive interview plan based on document analysis.
        
        Args:
            resume_analysis: Structured resume analysis
            job_analysis: Structured job description analysis  
            candidate_match: Candidate-job matching analysis
            interview_duration: Total interview duration in minutes
            custom_requirements: Additional custom requirements
            
        Returns:
            ProcessingResult with complete interview plan
        """
        
        start_time = time.time()
        self.stats["plans_generated"] += 1
        
        try:
            self.logger.info("Starting interview plan generation")
            
            # Set default duration
            duration = interview_duration or self.config.DEFAULT_INTERVIEW_DURATION
            
            # Generate interview strategy
            interview_strategy = await self._generate_interview_strategy(
                resume_analysis, job_analysis, candidate_match, custom_requirements
            )
            # save to file
            with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_strategy_{datetime.now().strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(interview_strategy, indent=2, default=str))
            print(f"In generate_interview_plan method of InterviewPlanningService: Interview strategy saved to file")
            # Create interview sections
            interview_sections = await self._create_interview_sections(
                interview_strategy, resume_analysis, job_analysis, duration
            )
            # save to file
            with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_sections_{datetime.now().strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(interview_sections, indent=2, default=str))
            print(f"In generate_interview_plan method of InterviewPlanningService: Interview sections saved to file")
            # Generate detailed questions for each section
            for section in interview_sections:
                await self._generate_section_questions(
                    section, interview_strategy, resume_analysis, job_analysis
                )
            
            # Create complete interview plan
            interview_plan = InterviewPlan(
                resume_analysis=resume_analysis,
                job_analysis=job_analysis,
                candidate_match=candidate_match,
                interview_sections=interview_sections,
                total_estimated_duration_minutes=duration,
                interview_objectives=interview_strategy["objectives"],
                key_focus_areas=interview_strategy["focus_areas"],
                evaluation_priorities=interview_strategy["evaluation_priorities"],
                potential_red_flags=interview_strategy["red_flags"],
                clarification_needed=interview_strategy["clarifications"],
                interviewer_notes=interview_strategy["interviewer_notes"],
                recommended_follow_up_areas=interview_strategy["follow_up_areas"]
            )
            # save to file
            # with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_plan_{datetime.now().strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
            #     f.write(json.dumps(interview_plan.model_dump(mode='json'), indent=2))
            # print(f"In generate_interview_plan method of InterviewPlanningService: Interview plan saved to file")
            # Calculate statistics
            total_questions = sum(len(section.questions) for section in interview_sections)
            self.stats["custom_questions_generated"] += total_questions
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_generations"] += 1
            self._update_average_questions()
            
            self.logger.info(f"Interview plan generated successfully in {processing_time:.2f}s")
            self.logger.info(f"Plan contains {len(interview_sections)} sections with {total_questions} questions")
            
            return ProcessingResult(
                success=True,
                result_data=interview_plan.dict(),
                processing_time_seconds=processing_time,
                confidence_score=0.9,  # High confidence for structured generation
                metadata={
                    "total_sections": len(interview_sections),
                    "total_questions": total_questions,
                    "interview_duration": duration,
                    "strategy_used": interview_strategy["strategy_type"],
                    "customizations_applied": bool(custom_requirements)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_generations"] += 1
            self.logger.error(f"Interview plan generation failed after {processing_time:.2f}s: {e}")
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    async def _generate_interview_strategy(self,
                                         resume_analysis: ResumeAnalysis,
                                         job_analysis: JobDescriptionAnalysis,
                                         candidate_match: Optional[CandidateMatch],
                                         custom_requirements: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate interview strategy based on analysis data."""
        
        # Build context for AI strategy generation
        context_prompt = f"""CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

Based on the following candidate and job analysis, generate an interview strategy:

CANDIDATE BACKGROUND:
- Technical Skills: {', '.join([skill.skill_name for skill in resume_analysis.technical_skills[:10]])}
- Experience Level: {resume_analysis.career_progression.total_experience_years if resume_analysis.career_progression else 'Unknown'} years
- Key Strengths: {', '.join(resume_analysis.strengths[:5])}
- Potential Gaps: {', '.join(resume_analysis.potential_gaps[:3])}

JOB REQUIREMENTS:
- Position: {job_analysis.job_title}
- Seniority: {job_analysis.role_seniority} {job_analysis.role_type}
- Key Technical Requirements: {', '.join([req.requirement for req in job_analysis.technical_requirements[:5]])}
- Critical Success Factors: {', '.join(job_analysis.critical_success_factors[:3])}

MATCH ANALYSIS:
{f'- Overall Match Score: {candidate_match.overall_match_score:.1%}' if candidate_match else '- Match analysis not available'}
{f'- Strong Matches: {", ".join(candidate_match.strong_matches[:3])}' if candidate_match and candidate_match.strong_matches else ''}
{f'- Skill Gaps: {", ".join(candidate_match.skill_gaps[:3])}' if candidate_match and candidate_match.skill_gaps else ''}

Generate an interview strategy that includes:
1. Primary interview objectives (3-5 key goals)
2. Key focus areas to explore in depth
3. Evaluation priorities for this specific candidate-role combination
4. Potential red flags to watch for
5. Areas that need clarification or deeper exploration
6. Interviewer notes and recommendations
7. Follow-up areas for future rounds

Format as JSON with these keys: objectives, focus_areas, evaluation_priorities, red_flags, clarifications, interviewer_notes, follow_up_areas, strategy_type"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert interview strategist. Generate comprehensive interview strategies based on candidate and job analysis. CRITICAL: Use ONLY standard ASCII characters - no Unicode symbols, checkmarks, bullet points, or special characters. Use simple text formatting with regular quotes and apostrophes."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            
            # Parse strategy from response
            strategy_content = response.choices[0].message.content
            
            try:
                # Try to extract JSON from response
                strategy_data = self._extract_json_from_response(strategy_content)
            except:
                # Fallback to basic strategy
                strategy_data = self._create_fallback_strategy(resume_analysis, job_analysis)
            
            return strategy_data
            
        except Exception as e:
            self.logger.warning(f"AI strategy generation failed: {e}, using fallback")
            return self._create_fallback_strategy(resume_analysis, job_analysis)
    
    def _create_fallback_strategy(self, resume_analysis: ResumeAnalysis, job_analysis: JobDescriptionAnalysis) -> Dict[str, Any]:
        """Create fallback strategy if AI generation fails."""
        
        return {
            "objectives": [
                "Assess technical competency for the role",
                "Evaluate cultural fit and communication skills", 
                "Validate experience claims from resume",
                "Determine motivation and career goals"
            ],
            "focus_areas": [
                skill.skill_name for skill in resume_analysis.technical_skills[:3]
            ] + ["Problem-solving approach", "Team collaboration"],
            "evaluation_priorities": [
                "Technical depth in core skills",
                "Experience relevance to role requirements",
                "Communication and interpersonal skills"
            ],
            "red_flags": [
                "Inconsistencies in experience timeline",
                "Lack of depth in claimed skills",
                "Poor communication or attitude"
            ],
            "clarifications": [
                "Verify technical skill proficiency levels",
                "Understand reasons for job changes",
                "Clarify availability and salary expectations"
            ],
            "interviewer_notes": [
                "Focus on behavioral examples for soft skills assessment",
                "Ask for specific technical examples and implementations",
                "Pay attention to enthusiasm and cultural fit indicators"
            ],
            "follow_up_areas": [
                "Technical deep-dive with team members",
                "Reference checks with previous employers",
                "Culture fit assessment with potential teammates"
            ],
            "strategy_type": "standard_comprehensive"
        }
    
    async def _create_interview_sections(self,
                                       strategy: Dict[str, Any],
                                       resume_analysis: ResumeAnalysis,
                                       job_analysis: JobDescriptionAnalysis,
                                       total_duration: int) -> List[InterviewSection]:
        """Create structured interview sections based on strategy."""
        
        sections = []
        
        # Calculate time allocations based on total duration
        time_allocations = self._calculate_time_allocations(total_duration)
        
        # Opening section
        opening_section = InterviewSection(
            phase=InterviewPhase.OPENING,
            section_name="Opening & Background",
            description="Initial conversation to make candidate comfortable and understand their background",
            objectives=self.question_frameworks["opening"]["objectives"],
            estimated_duration_minutes=time_allocations["opening"],
            questions=[],  # Will be populated later
            key_evaluation_points=[
                "Communication clarity",
                "Enthusiasm for role",
                "Professional presentation",
                "Cultural fit indicators"
            ],
            success_criteria=[
                "Candidate appears comfortable and engaged",
                "Clear articulation of interest in role",
                "Professional and positive demeanor"
            ]
        )
        sections.append(opening_section)
        
        # Technical section
        technical_section = InterviewSection(
            phase=InterviewPhase.TECHNICAL,
            section_name="Technical Assessment",
            description="Evaluate technical skills and problem-solving approach",
            objectives=self.question_frameworks["technical"]["objectives"],
            estimated_duration_minutes=time_allocations["technical"],
            questions=[],
            key_evaluation_points=[
                "Technical accuracy and depth",
                "Problem-solving methodology",
                "Code quality and best practices awareness",
                "Technology stack alignment"
            ],
            success_criteria=[
                "Demonstrates required technical competencies",
                "Shows clear problem-solving approach",
                "Articulates technical concepts effectively"
            ]
        )
        sections.append(technical_section)
        
        # Experience section
        experience_section = InterviewSection(
            phase=InterviewPhase.EXPERIENCE,
            section_name="Professional Experience",
            description="Deep dive into work history and achievements",
            objectives=self.question_frameworks["experience"]["objectives"],
            estimated_duration_minutes=time_allocations["experience"],
            questions=[],
            key_evaluation_points=[
                "Experience relevance to role",
                "Achievement impact and scale",
                "Leadership and collaboration skills",
                "Career progression logic"
            ],
            success_criteria=[
                "Provides concrete examples of achievements",
                "Shows progression in responsibilities",
                "Demonstrates relevant experience depth"
            ]
        )
        sections.append(experience_section)
        
        # Behavioral section
        behavioral_section = InterviewSection(
            phase=InterviewPhase.BEHAVIORAL,
            section_name="Behavioral & Cultural Fit",
            description="Assess soft skills, work style, and cultural alignment",
            objectives=self.question_frameworks["behavioral"]["objectives"],
            estimated_duration_minutes=time_allocations["behavioral"],
            questions=[],
            key_evaluation_points=[
                "Cultural fit with team and company",
                "Emotional intelligence and self-awareness",
                "Conflict resolution and communication skills",
                "Growth mindset and adaptability"
            ],
            success_criteria=[
                "Aligns with company values and culture",
                "Shows self-awareness and emotional maturity",
                "Demonstrates effective communication skills"
            ]
        )
        sections.append(behavioral_section)
        
        # Closing section
        closing_section = InterviewSection(
            phase=InterviewPhase.CLOSING,
            section_name="Closing & Next Steps",
            description="Answer questions and assess continued interest",
            objectives=self.question_frameworks["closing"]["objectives"],
            estimated_duration_minutes=time_allocations["closing"],
            questions=[],
            key_evaluation_points=[
                "Quality of questions asked",
                "Continued interest in role",
                "Salary and logistics alignment",
                "Professional closing impression"
            ],
            success_criteria=[
                "Asks thoughtful questions about role/company",
                "Maintains enthusiasm throughout interview",
                "Clear on expectations and next steps"
            ]
        )
        sections.append(closing_section)
        
        return sections
    
    def _calculate_time_allocations(self, total_duration: int) -> Dict[str, int]:
        """Calculate time allocation for each interview section."""
        
        # Base allocations (as percentages)
        base_allocations = {
            "opening": 0.10,    # 10%
            "technical": 0.35,  # 35% 
            "experience": 0.25, # 25%
            "behavioral": 0.25, # 25%
            "closing": 0.05     # 5%
        }
        
        # Calculate actual minutes
        allocations = {}
        for phase, percentage in base_allocations.items():
            allocations[phase] = int(total_duration * percentage)
        
        # Ensure minimum durations (Pydantic validation requires >= 5 minutes)
        min_duration = 5
        for phase in allocations:
            if allocations[phase] < min_duration:
                allocations[phase] = min_duration
        
        # Adjust for any rounding issues and ensure total matches
        total_allocated = sum(allocations.values())
        if total_allocated != total_duration:
            # Add difference to technical section (usually largest)
            allocations["technical"] += (total_duration - total_allocated)
            
            # If technical section becomes too small, redistribute
            if allocations["technical"] < min_duration:
                # Redistribute from other sections to maintain minimums
                excess = min_duration - allocations["technical"]
                allocations["technical"] = min_duration
                
                # Reduce from other sections proportionally
                for phase in ["experience", "behavioral", "opening"]:
                    if allocations[phase] > min_duration:
                        reduction = min(excess, allocations[phase] - min_duration)
                        allocations[phase] -= reduction
                        excess -= reduction
                        if excess <= 0:
                            break
        
        return allocations
    
    async def _generate_section_questions(self,
                                        section: InterviewSection,
                                        strategy: Dict[str, Any],
                                        resume_analysis: ResumeAnalysis,
                                        job_analysis: JobDescriptionAnalysis):
        """Generate specific questions for an interview section."""
        
        # Build context for question generation
        question_prompt = f"""CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

Generate {self.config.QUESTIONS_PER_SECTION} interview questions for the {section.section_name} section.

SECTION DETAILS:
- Phase: {section.phase.value}
- Duration: {section.estimated_duration_minutes} minutes
- Objectives: {', '.join(section.objectives)}

CANDIDATE CONTEXT:
- Technical Skills: {', '.join([skill.skill_name for skill in resume_analysis.technical_skills[:8]])}
- Recent Experience: {resume_analysis.work_experience[0].job_title if resume_analysis.work_experience else 'Not available'} at {resume_analysis.work_experience[0].company_name if resume_analysis.work_experience else 'Unknown'}
- Strengths: {', '.join(resume_analysis.strengths[:3])}

JOB CONTEXT:
- Position: {job_analysis.job_title}
- Key Requirements: {', '.join([req.requirement for req in job_analysis.technical_requirements[:5]])}
- Success Factors: {', '.join(job_analysis.critical_success_factors[:3])}

INTERVIEW STRATEGY:
- Focus Areas: {', '.join(strategy['focus_areas'][:5])}
- Areas to Probe: {', '.join(strategy.get('clarifications', [])[:3])}

Generate questions that:
1. Are appropriate for the interview phase
2. Target the candidate's specific background
3. Assess relevant skills for the job requirements
4. Allow for follow-up and deep exploration
5. Take 5-10 minutes each to answer (minimum 5 minutes per question)

CRITICAL DATA STRUCTURE RULES:
- estimated_time_minutes MUST be >= 5 (Pydantic validation requirement)
- skill_focus and evaluation_criteria MUST be arrays, never null
- Use empty arrays [] for missing optional arrays
- difficulty_level must be 1-5 integer

VALID QUESTION TYPES (use exactly these values):
- "open_ended" - for general questions
- "technical" - for technical skill assessment
- "behavioral" - for behavior and experience questions
- "situational" - for scenario-based questions
- "follow_up" - for follow-up questions
- "clarification" - for clarification questions

Format as JSON array with objects containing: question_text, question_type (one of the valid types above), difficulty_level (1-5), estimated_time_minutes (>=5), skill_focus (array), evaluation_criteria (array)"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert interview question generator specializing in {section.phase.value} questions. CRITICAL: Use ONLY standard ASCII characters - no Unicode symbols, checkmarks, bullet points, or special characters. Use simple text formatting with regular quotes and apostrophes."},
                    {"role": "user", "content": question_prompt}
                ],
                max_tokens=2000,
                temperature=0.5
            )
            
            # Parse questions from response
            questions_content = response.choices[0].message.content
            
            try:
                questions_data = self._extract_json_from_response(questions_content)
                
                # Convert to Question objects
                questions = []
                for q_data in questions_data:
                    # Map question type to valid enum values
                    question_type_str = q_data.get("question_type", "open_ended")
                    question_type = self._map_question_type(question_type_str)
                    
                    question = Question(
                        question_text=q_data.get("question_text", ""),
                        question_type=question_type,
                        phase=section.phase,
                        difficulty_level=q_data.get("difficulty_level", 3),
                        estimated_time_minutes=q_data.get("estimated_time_minutes", 5),
                        skill_focus=q_data.get("skill_focus", []),
                        evaluation_criteria=q_data.get("evaluation_criteria", [])
                    )
                    questions.append(question)
                
                section.questions = questions
                
            except Exception as e:
                self.logger.warning(f"Failed to parse generated questions: {e}, using fallback")
                self._create_fallback_questions(section, resume_analysis, job_analysis)
                
        except Exception as e:
            self.logger.warning(f"Question generation failed: {e}, using fallback")
            self._create_fallback_questions(section, resume_analysis, job_analysis)
    
    def _create_fallback_questions(self,
                                 section: InterviewSection,
                                 resume_analysis: ResumeAnalysis,
                                 job_analysis: JobDescriptionAnalysis):
        """Create fallback questions if AI generation fails."""
        
        framework = self.question_frameworks.get(section.phase.value, {})
        sample_questions = framework.get("sample_questions", [])
        question_types = framework.get("question_types", [QuestionType.OPEN_ENDED])
        
        questions = []
        
        for i, q_text in enumerate(sample_questions[:self.config.QUESTIONS_PER_SECTION]):
            question = Question(
                question_text=q_text,
                question_type=question_types[i % len(question_types)],
                phase=section.phase,
                difficulty_level=3,
                estimated_time_minutes=section.estimated_duration_minutes // self.config.QUESTIONS_PER_SECTION,
                skill_focus=["General assessment"],
                evaluation_criteria=["Response quality", "Communication clarity"]
            )
            questions.append(question)
        
        section.questions = questions
    
    def _extract_json_from_response(self, response_content: str) -> Any:
        """Extract JSON from AI response."""
        
        import re
        import json
        
        try:
            # Try direct JSON parsing
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to find JSON within response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON array or object
            json_match = re.search(r'[\[\{].*[\]\}]', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError("No valid JSON found in response")
    
    def _map_question_type(self, question_type_str: str) -> QuestionType:
        """Map question type strings to valid QuestionType enum values."""
        # Normalize the string
        normalized = question_type_str.lower().strip()
        
        # Direct mappings
        type_mapping = {
            "open_ended": QuestionType.OPEN_ENDED,
            "technical": QuestionType.TECHNICAL,
            "behavioral": QuestionType.BEHAVIORAL,
            "situational": QuestionType.SITUATIONAL,
            "follow_up": QuestionType.FOLLOW_UP,
            "clarification": QuestionType.CLARIFICATION,
            # Handle common variations
            "opening": QuestionType.OPEN_ENDED,
            "closing": QuestionType.OPEN_ENDED,
            "experience": QuestionType.BEHAVIORAL,
            "general": QuestionType.OPEN_ENDED,
            "problem_solving": QuestionType.SITUATIONAL,
            "scenario": QuestionType.SITUATIONAL,
            "background": QuestionType.OPEN_ENDED,
            "motivation": QuestionType.BEHAVIORAL,
            "teamwork": QuestionType.BEHAVIORAL,
            "leadership": QuestionType.BEHAVIORAL,
            "communication": QuestionType.BEHAVIORAL,
            "culture_fit": QuestionType.BEHAVIORAL,
            "career_goals": QuestionType.BEHAVIORAL,
            "salary": QuestionType.OPEN_ENDED,
            "availability": QuestionType.OPEN_ENDED
        }
        
        # Return mapped type or default to open_ended
        return type_mapping.get(normalized, QuestionType.OPEN_ENDED)
    
    def _update_average_questions(self):
        """Update average questions per plan statistic."""
        if self.stats["successful_generations"] > 0:
            self.stats["average_questions_per_plan"] = (
                self.stats["custom_questions_generated"] / self.stats["successful_generations"]
            )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_generations"] / max(1, self.stats["plans_generated"])
            ),
            "average_processing_time": (
                self.stats["total_processing_time"] / max(1, self.stats["successful_generations"])
            ),
            "question_frameworks_available": len(self.question_frameworks),
            "service_status": "active"
        }


# Export main class
__all__ = ['InterviewPlanningService', 'InterviewPlanningConfig']
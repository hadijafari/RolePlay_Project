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
        ProcessingResult, SimplifiedInterviewPlan, SimplifiedInterviewSection, SimplifiedQuestion,
        SimplifiedResumeAnalysis
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
    
    # Interview structure settings (durations are informational only; no enforcement)
    DEFAULT_INTERVIEW_DURATION = 60  # minutes (not enforced)
    
   # set it to 30 minutes in total
    # DEFAULT_INTERVIEW_DURATION = 30  # minutes
    # OPENING_DURATION = 3    # minutes
    # TECHNICAL_DURATION = 8  # minutes  
    # EXPERIENCE_DURATION = 8 # minutes
    # BEHAVIORAL_DURATION = 8 # minutes
    # CLOSING_DURATION = 3    # minutes


    # Question generation settings
    QUESTIONS_PER_SECTION = 5  # default fallback
    BACKUP_QUESTIONS_PER_SECTION = 3
    # Per-section targets to reach 25 total (informational; no time limits)
    QUESTIONS_PER_SECTION_BY_NAME = {
        "Opening & Introduction": 3,
        "Warm-up": 3,
        "Core Questions / Main Body": 12,
        "Challenging or Sensitive Topics": 3,
        "Closing Questions": 2,
        "Wrap-up": 2
    }
    
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
            "opening_intro": {
                "objectives": [
                    "Build rapport and set expectations",
                    "Introduce interviewer and purpose",
                    "Obtain consent for recording if applicable",
                    "Outline interview process and timeline"
                ],
                "question_types": [QuestionType.OPEN_ENDED],
                "sample_questions": [
                    "To begin, could you confirm you're comfortable proceeding with this interview?",
                    "I'll briefly outline the flow: opening, warm-up, core topics, a few challenges, and closing. Does that work for you?",
                    "In one or two sentences, what attracted you to this role?"
                ]
            },
            "warm_up": {
                "objectives": [
                    "Help the candidate relax",
                    "Encourage natural conversation",
                    "Gather high-level context"
                ],
                "question_types": [QuestionType.OPEN_ENDED, QuestionType.BEHAVIORAL],
                "sample_questions": [
                    "What recent project are you most proud of, and why?",
                    "Which tools or environments do you feel most at home with?",
                    "If you had to explain your core expertise to a non-technical person, how would you do it?"
                ]
            },
            "core_main": {
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
            "challenging": {
                "objectives": [
                    "Explore constraints, trade-offs, and risks",
                    "Discuss sensitive or difficult topics respectfully",
                    "Understand decision-making under pressure"
                ],
                "question_types": [QuestionType.SITUATIONAL, QuestionType.BEHAVIORAL],
                "sample_questions": [
                    "Tell me about a time you faced a difficult constraint or safety/regulatory requirement. How did it shape your solution?",
                    "Describe a tough trade-off you made (performance vs. cost, latency vs. accuracy, etc.).",
                    "Share an instance where something went wrong in production. How did you handle it and what changed after?"
                ]
            },
            "closing_questions": {
                "objectives": [
                    "Give the candidate space to add anything missing",
                    "Clarify remaining topics",
                    "Transition to next steps"
                ],
                "question_types": [QuestionType.OPEN_ENDED],
                "sample_questions": [
                    "Is there anything we haven't covered that you think is important?",
                    "Are there any areas you'd like to clarify or expand on?",
                    "What questions do you have for us about the role or team?"
                ]
            },
            "wrap_up": {
                "objectives": [
                    "Thank the candidate and explain next steps",
                    "Confirm contact details or follow-ups",
                    "Close on a positive, professional note"
                ],
                "question_types": [QuestionType.OPEN_ENDED],
                "sample_questions": [
                    "Thank you for your time today. We'll review and get back to you. Does that timeline work?",
                    "Do you have a preferred method of contact for follow-up?",
                    "Before we wrap up, is there anything else you'd like to add?"
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
            # Create interview sections (six-part structure; durations informational only)
            interview_sections = await self._create_interview_sections(
                interview_strategy, resume_analysis, job_analysis, duration
            )
            # save to file
            with open(f"D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\interview_sections_{datetime.now().strftime('%Y-%m-%d')}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(interview_sections, indent=2, default=str))
            print(f"In generate_interview_plan method of InterviewPlanningService: Interview sections saved to file")
            # Generate detailed questions for each section with de-duplication across sections
            previous_q_texts: List[str] = []
            for section in interview_sections:
                await self._generate_section_questions(
                    section, interview_strategy, resume_analysis, job_analysis, previous_q_texts
                )
                previous_q_texts.extend([q.question_text for q in section.questions])
            
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
        
        # 1) Opening / Introduction
        opening_section = InterviewSection(
            phase=InterviewPhase.OPENING,
            section_name="Opening & Introduction",
            description="Introduce the interview, obtain consent if applicable, build rapport, and set expectations",
            objectives=self.question_frameworks["opening_intro"]["objectives"],
            estimated_duration_minutes=time_allocations["opening_intro"],
            questions=[],  # Will be populated later
            key_evaluation_points=[
                "Rapport and professionalism",
                "Clarity of communication",
                "Motivation and alignment"
            ],
            success_criteria=[
                "Candidate appears comfortable and engaged",
                "Consent obtained if applicable",
                "Clear understanding of process"
            ]
        )
        sections.append(opening_section)
        
        # 2) Warm-up Questions
        warmup_section = InterviewSection(
            phase=InterviewPhase.OPENING,
            section_name="Warm-up",
            description="Easy, open-ended questions to help the candidate relax and provide context",
            objectives=self.question_frameworks["warm_up"]["objectives"],
            estimated_duration_minutes=time_allocations["warm_up"],
            questions=[],
            key_evaluation_points=[
                "Comfort level",
                "High-level coherence",
                "Relevance to role"
            ],
            success_criteria=[
                "Conversation flows naturally",
                "Candidate provides concise, relevant context"
            ]
        )
        sections.append(warmup_section)
        
        # 3) Core Questions / Main Body (technical focus)
        core_section = InterviewSection(
            phase=InterviewPhase.TECHNICAL,
            section_name="Core Questions / Main Body",
            description="Primary technical and role-related assessment; progress from broad to specific",
            objectives=self.question_frameworks["core_main"]["objectives"],
            estimated_duration_minutes=time_allocations["core_main"],
            questions=[],
            key_evaluation_points=[
                "Technical accuracy and depth",
                "Problem-solving methodology",
                "Architecture/design trade-offs",
                "Technology stack alignment"
            ],
            success_criteria=[
                "Demonstrates required technical competencies",
                "Explains reasoning and constraints clearly",
                "Connects experience to role requirements"
            ]
        )
        sections.append(core_section)
        
        # 4) Challenging or Sensitive Topics
        challenging_section = InterviewSection(
            phase=InterviewPhase.SITUATIONAL,
            section_name="Challenging or Sensitive Topics",
            description="Explore difficult constraints, failures, and risk/impact scenarios respectfully",
            objectives=self.question_frameworks["challenging"]["objectives"],
            estimated_duration_minutes=time_allocations["challenging"],
            questions=[],
            key_evaluation_points=[
                "Decision-making under pressure",
                "Risk awareness and mitigation",
                "Ownership and learning from mistakes"
            ],
            success_criteria=[
                "Gives candid, reflective answers",
                "Shows mature judgment and accountability"
            ]
        )
        sections.append(challenging_section)
        
        # 5) Closing Questions
        closing_q_section = InterviewSection(
            phase=InterviewPhase.CLOSING,
            section_name="Closing Questions",
            description="Give the candidate a chance to add anything and ask questions",
            objectives=self.question_frameworks["closing_questions"]["objectives"],
            estimated_duration_minutes=time_allocations["closing_questions"],
            questions=[],
            key_evaluation_points=[
                "Completeness of coverage",
                "Candidate questions",
                "Interest alignment"
            ],
            success_criteria=[
                "Candidate asked thoughtful questions",
                "Any critical gaps were addressed"
            ]
        )
        sections.append(closing_q_section)

        # 6) Wrap-up
        wrapup_section = InterviewSection(
            phase=InterviewPhase.CLOSING,
            section_name="Wrap-up",
            description="Thank them, explain next steps, confirm contact preferences, and close",
            objectives=self.question_frameworks["wrap_up"]["objectives"],
            estimated_duration_minutes=time_allocations["wrap_up"],
            questions=[],
            key_evaluation_points=[
                "Professional courtesy",
                "Clarity on next steps"
            ],
            success_criteria=[
                "Candidate leaves with clear expectations",
                "Positive closing tone"
            ]
        )
        sections.append(wrapup_section)
        
        return sections
    
    def _calculate_time_allocations(self, total_duration: int) -> Dict[str, int]:
        """Calculate time allocation for each interview section."""
        
        # Base allocations (as percentages) — informational only
        base_allocations = {
            "opening_intro": 0.08,
            "warm_up": 0.10,
            "core_main": 0.55,
            "challenging": 0.15,
            "closing_questions": 0.07,
            "wrap_up": 0.05
        }
        
        # Calculate actual minutes
        allocations = {}
        for phase, percentage in base_allocations.items():
            allocations[phase] = int(total_duration * percentage)
        
        # Ensure minimum durations for model validation (informational only)
        min_duration = 5
        for phase in allocations:
            if allocations[phase] < min_duration:
                allocations[phase] = min_duration
        
        # Adjust for any rounding issues and ensure total matches
        total_allocated = sum(allocations.values())
        if total_allocated != total_duration:
            # Add difference to core_main section (usually largest)
            allocations["core_main"] += (total_duration - total_allocated)
            
            # If technical section becomes too small, redistribute
            if allocations["core_main"] < min_duration:
                # Redistribute from other sections to maintain minimums
                excess = min_duration - allocations["core_main"]
                allocations["core_main"] = min_duration
                
                # Reduce from other sections proportionally
                for phase in ["closing_questions", "warm_up", "opening_intro"]:
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
                                        job_analysis: JobDescriptionAnalysis,
                                        previous_questions: List[str]):
        """Generate specific questions for an interview section."""
        
        # Build context for question generation
        target_count = self._get_questions_target_for_section(section)

        prev_q_block = "\n".join([f"- {q}" for q in previous_questions[-20:]]) if previous_questions else "(none)"

        question_prompt = f"""CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

Generate {target_count} interview questions for the {section.section_name} section.

SECTION DETAILS:
- Phase: {section.phase.value}
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

PREVIOUSLY PLANNED QUESTIONS (avoid repeating these; go a layer deeper instead of restating):
{prev_q_block}

Generate questions that:
1. Are appropriate for the interview phase
2. Target the candidate's specific background
3. Assess relevant skills for the job requirements
4. Allow for follow-up and deeper exploration
5. Avoid redundancy across sections. Do NOT re-ask high-level "describe your experience with X" if already implied by the resume; ask for specific aspects: constraints, interfaces, debugging approach, rationale, edge cases, trade-offs. Keep depth intermediate (not ultra-expert).

CRITICAL DATA STRUCTURE RULES:
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

Format as JSON array with objects containing: question_text, question_type (one of the valid types above), difficulty_level (1-5), estimated_time_minutes (>=1), skill_focus (array), evaluation_criteria (array)"""
        
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
                        estimated_time_minutes=max(1, q_data.get("estimated_time_minutes", 5)),
                        skill_focus=q_data.get("skill_focus", []),
                        evaluation_criteria=q_data.get("evaluation_criteria", [])
                    )
                    # De-duplicate against previous sections
                    if not self._is_duplicate_question(question.question_text, previous_questions):
                        questions.append(question)
                
                # Trim to target count if over-generated
                section.questions = questions[:target_count]
                
            except Exception as e:
                self.logger.warning(f"Failed to parse generated questions: {e}, using fallback")
                self._create_fallback_questions(section, resume_analysis, job_analysis)
                
        except Exception as e:
            self.logger.warning(f"Question generation failed: {e}, using fallback")
            self._create_fallback_questions(section, resume_analysis, job_analysis)

    def _get_questions_target_for_section(self, section: InterviewSection) -> int:
        """Determine how many questions to generate for a given section name."""
        return self.config.QUESTIONS_PER_SECTION_BY_NAME.get(section.section_name, self.config.QUESTIONS_PER_SECTION)

    def _is_duplicate_question(self, question_text: str, previous_questions: List[str]) -> bool:
        """Simple semantic de-duplication: lowercase match or high token overlap with prior questions."""
        qt = question_text.strip().lower()
        if not qt:
            return True
        for prev in previous_questions:
            pv = prev.strip().lower()
            if pv == qt:
                return True
            # token overlap
            qt_tokens = set([t for t in qt.split() if len(t) > 2])
            pv_tokens = set([t for t in pv.split() if len(t) > 2])
            if qt_tokens and (len(qt_tokens & pv_tokens) / len(qt_tokens)) > 0.8:
                return True
        return False
    
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
    
    async def generate_simplified_interview_plan_from_simplified_resume(self,
                                                                      simplified_resume_analysis: SimplifiedResumeAnalysis,
                                                                      job_analysis: JobDescriptionAnalysis,
                                                                      candidate_match: Optional[CandidateMatch] = None,
                                                                      interview_duration: int = None,
                                                                      custom_requirements: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Generate a simplified interview plan based on simplified resume analysis.
        
        Args:
            simplified_resume_analysis: Simplified resume analysis
            job_analysis: Structured job description analysis  
            candidate_match: Candidate-job matching analysis
            interview_duration: Total interview duration in minutes
            custom_requirements: Additional custom requirements
            
        Returns:
            ProcessingResult with simplified interview plan
        """
        
        start_time = time.time()
        self.stats["plans_generated"] += 1
        
        try:
            self.logger.info("Starting simplified interview plan generation from simplified resume")
            
            # Set default duration
            duration = interview_duration or self.config.DEFAULT_INTERVIEW_DURATION
            
            # Generate interview strategy using simplified data
            interview_strategy = await self._generate_interview_strategy_from_simplified_resume(
                simplified_resume_analysis, job_analysis, candidate_match, custom_requirements
            )
            
            # Create interview sections (six-part structure; durations informational only)
            interview_sections = await self._create_interview_sections_from_simplified_resume(
                interview_strategy, simplified_resume_analysis, job_analysis, duration
            )
            
            # Generate detailed questions for each section with de-duplication across sections
            previous_q_texts: List[str] = []
            for section in interview_sections:
                await self._generate_section_questions_from_simplified_resume(
                    section, interview_strategy, simplified_resume_analysis, job_analysis, previous_q_texts
                )
                previous_q_texts.extend([q.question_text for q in section.questions])
            
            # Create complete simplified interview plan
            interview_plan = SimplifiedInterviewPlan(
                candidate_match=candidate_match,
                interview_sections=interview_sections,
                total_estimated_duration_minutes=duration
            )
            
            # Calculate statistics
            total_questions = sum(len(section.questions) for section in interview_sections)
            self.stats["custom_questions_generated"] += total_questions
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_generations"] += 1
            self._update_average_questions()
            
            self.logger.info(f"Simplified interview plan generated successfully in {processing_time:.2f}s")
            self.logger.info(f"Plan contains {len(interview_sections)} sections with {total_questions} questions")
            
            return ProcessingResult(
                success=True,
                result_data=interview_plan.dict(),
                processing_time_seconds=processing_time,
                confidence_score=0.8,  # Slightly lower confidence due to simplified data
                metadata={
                    "total_sections": len(interview_sections),
                    "total_questions": total_questions,
                    "interview_duration": duration,
                    "plan_type": "simplified_from_simplified_resume"
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_generations"] += 1
            self.logger.error(f"Simplified interview plan generation failed after {processing_time:.2f}s: {e}")
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )

    async def generate_simplified_interview_plan(self,
                                               resume_analysis: ResumeAnalysis,
                                               job_analysis: JobDescriptionAnalysis,
                                               candidate_match: Optional[CandidateMatch] = None,
                                               interview_duration: int = None,
                                               custom_requirements: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Generate a simplified interview plan based on document analysis.
        
        Args:
            resume_analysis: Structured resume analysis
            job_analysis: Structured job description analysis  
            candidate_match: Candidate-job matching analysis
            interview_duration: Total interview duration in minutes
            custom_requirements: Additional custom requirements
            
        Returns:
            ProcessingResult with simplified interview plan
        """
        
        start_time = time.time()
        self.stats["plans_generated"] += 1
        
        try:
            self.logger.info("Starting simplified interview plan generation")
            
            # Generate the complex interview plan first
            complex_plan_result = await self.generate_interview_plan(
                resume_analysis, job_analysis, candidate_match, interview_duration, custom_requirements
            )
            
            if not complex_plan_result.success:
                return complex_plan_result
            
            # Extract the complex plan
            complex_plan_data = complex_plan_result.result_data
            complex_plan = InterviewPlan(**complex_plan_data)
            
            # Convert to simplified plan
            simplified_plan = self.convert_to_simplified_plan(complex_plan)
            
            # Calculate statistics
            total_questions = sum(len(section.questions) for section in simplified_plan.interview_sections)
            self.stats["custom_questions_generated"] += total_questions
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_generations"] += 1
            self._update_average_questions()
            
            self.logger.info(f"Simplified interview plan generated successfully in {processing_time:.2f}s")
            self.logger.info(f"Plan contains {len(simplified_plan.interview_sections)} sections with {total_questions} questions")
            
            return ProcessingResult(
                success=True,
                result_data=simplified_plan.dict(),
                processing_time_seconds=processing_time,
                confidence_score=0.9,  # High confidence for structured generation
                metadata={
                    "total_sections": len(simplified_plan.interview_sections),
                    "total_questions": total_questions,
                    "interview_duration": simplified_plan.total_estimated_duration_minutes,
                    "plan_type": "simplified"
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_generations"] += 1
            self.logger.error(f"Simplified interview plan generation failed after {processing_time:.2f}s: {e}")
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )

    def convert_to_simplified_plan(self, interview_plan: InterviewPlan) -> SimplifiedInterviewPlan:
        """Convert a complex InterviewPlan to a SimplifiedInterviewPlan."""
        try:
            # Convert interview sections
            simplified_sections = []
            for section in interview_plan.interview_sections:
                # Convert questions
                simplified_questions = []
                for question in section.questions:
                    simplified_question = SimplifiedQuestion(
                        question_text=question.question_text
                    )
                    simplified_questions.append(simplified_question)
                
                # Create simplified section
                simplified_section = SimplifiedInterviewSection(
                    phase=section.phase,
                    section_name=section.section_name,
                    description=section.description,
                    estimated_duration_minutes=section.estimated_duration_minutes,
                    questions=simplified_questions
                )
                simplified_sections.append(simplified_section)
            
            # Create simplified interview plan
            simplified_plan = SimplifiedInterviewPlan(
                plan_id=interview_plan.plan_id,
                created_timestamp=interview_plan.created_timestamp,
                candidate_match=interview_plan.candidate_match,
                interview_sections=simplified_sections,
                total_estimated_duration_minutes=interview_plan.total_estimated_duration_minutes
            )
            
            return simplified_plan
            
        except Exception as e:
            self.logger.error(f"Failed to convert interview plan to simplified format: {e}")
            raise
    
    async def _generate_interview_strategy_from_simplified_resume(self,
                                                               simplified_resume_analysis: SimplifiedResumeAnalysis,
                                                               job_analysis: JobDescriptionAnalysis,
                                                               candidate_match: Optional[CandidateMatch],
                                                               custom_requirements: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate interview strategy based on simplified resume analysis data."""
        
        # Build context for AI strategy generation
        context_prompt = f"""CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

Based on the following simplified candidate and job analysis, generate an interview strategy:

CANDIDATE BACKGROUND:
- Technical Skills: {', '.join([skill.skill_name for skill in simplified_resume_analysis.technical_skills[:10]])}
- Experience Level: {simplified_resume_analysis.total_experience_years} years
- Key Strengths: {', '.join(simplified_resume_analysis.key_strengths[:5])}

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
                strategy_data = self._create_fallback_strategy_from_simplified_resume(simplified_resume_analysis, job_analysis)
            
            return strategy_data
            
        except Exception as e:
            self.logger.warning(f"AI strategy generation failed: {e}, using fallback")
            return self._create_fallback_strategy_from_simplified_resume(simplified_resume_analysis, job_analysis)
    
    def _create_fallback_strategy_from_simplified_resume(self, simplified_resume_analysis: SimplifiedResumeAnalysis, job_analysis: JobDescriptionAnalysis) -> Dict[str, Any]:
        """Create fallback strategy if AI generation fails."""
        
        return {
            "objectives": [
                "Assess technical competency for the role",
                "Evaluate cultural fit and communication skills", 
                "Validate experience claims from resume",
                "Determine motivation and career goals"
            ],
            "focus_areas": [
                skill.skill_name for skill in simplified_resume_analysis.technical_skills[:3]
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
            "strategy_type": "simplified_comprehensive"
        }
    
    async def _create_interview_sections_from_simplified_resume(self,
                                                             strategy: Dict[str, Any],
                                                             simplified_resume_analysis: SimplifiedResumeAnalysis,
                                                             job_analysis: JobDescriptionAnalysis,
                                                             total_duration: int) -> List[SimplifiedInterviewSection]:
        """Create structured interview sections based on simplified resume strategy."""
        
        sections = []
        
        # Calculate time allocations based on total duration
        time_allocations = self._calculate_time_allocations(total_duration)
        
        # 1) Opening / Introduction
        opening_section = SimplifiedInterviewSection(
            phase=InterviewPhase.OPENING,
            section_name="Opening & Introduction",
            description="Introduce the interview, obtain consent if applicable, build rapport, and set expectations",
            estimated_duration_minutes=time_allocations["opening_intro"],
            questions=[]
        )
        sections.append(opening_section)
        
        # 2) Warm-up Questions
        warmup_section = SimplifiedInterviewSection(
            phase=InterviewPhase.OPENING,
            section_name="Warm-up",
            description="Easy, open-ended questions to help the candidate relax and provide context",
            estimated_duration_minutes=time_allocations["warm_up"],
            questions=[]
        )
        sections.append(warmup_section)
        
        # 3) Core Questions / Main Body (technical focus)
        core_section = SimplifiedInterviewSection(
            phase=InterviewPhase.TECHNICAL,
            section_name="Core Questions / Main Body",
            description="Primary technical and role-related assessment; progress from broad to specific",
            estimated_duration_minutes=time_allocations["core_main"],
            questions=[]
        )
        sections.append(core_section)
        
        # 4) Challenging or Sensitive Topics
        challenging_section = SimplifiedInterviewSection(
            phase=InterviewPhase.SITUATIONAL,
            section_name="Challenging or Sensitive Topics",
            description="Explore difficult constraints, failures, and risk/impact scenarios respectfully",
            estimated_duration_minutes=time_allocations["challenging"],
            questions=[]
        )
        sections.append(challenging_section)
        
        # 5) Closing Questions
        closing_q_section = SimplifiedInterviewSection(
            phase=InterviewPhase.CLOSING,
            section_name="Closing Questions",
            description="Give the candidate a chance to add anything and ask questions",
            estimated_duration_minutes=time_allocations["closing_questions"],
            questions=[]
        )
        sections.append(closing_q_section)
        
        # 6) Wrap-up
        wrapup_section = SimplifiedInterviewSection(
            phase=InterviewPhase.CLOSING,
            section_name="Wrap-up",
            description="Thank them, explain next steps, confirm contact preferences, and close",
            estimated_duration_minutes=time_allocations["wrap_up"],
            questions=[]
        )
        sections.append(wrapup_section)
        
        return sections
    
    async def _generate_section_questions_from_simplified_resume(self,
                                                              section: SimplifiedInterviewSection,
                                                              strategy: Dict[str, Any],
                                                              simplified_resume_analysis: SimplifiedResumeAnalysis,
                                                              job_analysis: JobDescriptionAnalysis,
                                                              previous_questions: List[str]):
        """Generate specific questions for an interview section using simplified resume data."""
        
        # Build context for question generation
        target_count = self._get_questions_target_for_section_name(section.section_name)

        prev_q_block = "\n".join([f"- {q}" for q in previous_questions[-20:]]) if previous_questions else "(none)"

        question_prompt = f"""CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

Generate {target_count} interview questions for the {section.section_name} section.

SECTION DETAILS:
- Phase: {section.phase.value}
- Description: {section.description}

CANDIDATE CONTEXT (SIMPLIFIED):
- Technical Skills: {', '.join([skill.skill_name for skill in simplified_resume_analysis.technical_skills[:8]])}
- Experience: {simplified_resume_analysis.total_experience_years} years
- Key Strengths: {', '.join(simplified_resume_analysis.key_strengths[:3])}

JOB CONTEXT:
- Position: {job_analysis.job_title}
- Key Requirements: {', '.join([req.requirement for req in job_analysis.technical_requirements[:5]])}
- Success Factors: {', '.join(job_analysis.critical_success_factors[:3])}

INTERVIEW STRATEGY:
- Focus Areas: {', '.join(strategy['focus_areas'][:5])}
- Areas to Probe: {', '.join(strategy.get('clarifications', [])[:3])}

PREVIOUSLY PLANNED QUESTIONS (avoid repeating these):
{prev_q_block}

Generate questions that:
1. Are appropriate for the interview phase
2. Target the candidate's specific background
3. Assess relevant skills for the job requirements
4. Allow for follow-up and deeper exploration
5. Avoid redundancy across sections

Format as JSON array with objects containing: question_text"""
        
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
                
                # Convert to SimplifiedQuestion objects
                questions = []
                for q_data in questions_data:
                    question = SimplifiedQuestion(
                        question_text=q_data.get("question_text", "")
                    )
                    # De-duplicate against previous sections
                    if not self._is_duplicate_question(question.question_text, previous_questions):
                        questions.append(question)
                
                # Limit to target count
                section.questions = questions[:target_count]
                
            except Exception as e:
                self.logger.warning(f"Failed to parse questions for {section.section_name}: {e}")
                # Use fallback questions
                section.questions = self._get_fallback_questions_for_section(section.section_name, target_count)
                
        except Exception as e:
            self.logger.warning(f"Question generation failed for {section.section_name}: {e}")
            # Use fallback questions
            section.questions = self._get_fallback_questions_for_section(section.section_name, target_count)
    
    def _get_questions_target_for_section_name(self, section_name: str) -> int:
        """Get target question count for section by name."""
        return self.config.QUESTIONS_PER_SECTION_BY_NAME.get(section_name, self.config.QUESTIONS_PER_SECTION)
    
    def _get_fallback_questions_for_section(self, section_name: str, target_count: int) -> List[SimplifiedQuestion]:
        """Get fallback questions for a section."""
        fallback_questions = {
            "Opening & Introduction": [
                "Tell me about yourself and your background.",
                "What interests you about this role?",
                "Why are you looking for a new opportunity?"
            ],
            "Warm-up": [
                "Can you walk me through your most recent project?",
                "What technologies have you been working with lately?",
                "How do you stay updated with industry trends?"
            ],
            "Core Questions / Main Body": [
                "Describe a challenging technical problem you solved recently.",
                "How do you approach debugging complex issues?",
                "Can you explain your experience with [specific technology]?",
                "What's your process for code review and quality assurance?",
                "How do you handle tight deadlines and competing priorities?"
            ],
            "Challenging or Sensitive Topics": [
                "Tell me about a time when a project didn't go as planned.",
                "How do you handle disagreements with team members?",
                "Describe a situation where you had to learn something completely new quickly."
            ],
            "Closing Questions": [
                "Is there anything else you'd like to tell us about your experience?",
                "Do you have any questions about the role or company?"
            ],
            "Wrap-up": [
                "Thank you for your time today. We'll be in touch soon.",
                "Do you have any final questions before we wrap up?"
            ]
        }
        
        questions = fallback_questions.get(section_name, ["Tell me about your experience."])
        return [SimplifiedQuestion(question_text=q) for q in questions[:target_count]]


# Export main class
__all__ = ['InterviewPlanningService', 'InterviewPlanningConfig']
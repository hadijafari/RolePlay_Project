"""
Interview Conductor Agent
Specialized agent for conducting structured interviews using properly injected context.
This agent fixes the critical context passing issue and enables dynamic interview management.
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
    from models.interview_models import (
        InterviewPlan, InterviewState, InterviewPhase, Question, 
        ResponseEvaluation, InterviewContext, QuestionType, FollowUpConfig
    )
    from services.context_injection_service import ContextInjectionService, create_interview_context
except ImportError as e:
    print(f"Interview Conductor Agent: Missing required modules: {e}")
    sys.exit(1)


class InterviewConductorAgent(BaseAgent):
    """
    Specialized agent for conducting structured interviews with proper context handling.
    
    This agent solves the critical context passing problem by:
    1. Using structured context injection instead of text flattening
    2. Maintaining interview state and progression
    3. Dynamically selecting questions based on responses
    4. Providing real-time evaluation and follow-up
    """
    
    def __init__(self, 
                 interview_plan: InterviewPlan,
                 name: str = "Interview Conductor",
                 model: str = "gpt-4o-mini"):
        """
        Initialize interview conductor agent with a specific interview plan.
        
        Args:
            interview_plan: Structured interview plan to follow
            name: Agent name
            model: OpenAI model to use
        """
        # Generate specialized instructions for interview conducting
        instructions = self._generate_interview_instructions(interview_plan)
        
        # Configure agent for interview context
        config = AgentConfig(
            name=name,
            instructions=instructions,
            model=model,
            timeout=45.0,  # Longer timeout for thoughtful responses
            max_history_length=100,  # Keep full interview history
            log_level="INFO",
            use_structured_context=True,  # CRITICAL: Enable structured context
            agent_type="interview"  # CRITICAL: Set agent type for proper context handling
        )
        
        # Initialize base agent
        super().__init__(config)
        
        # Interview-specific state management
        self.interview_plan = interview_plan
        self.interview_state = InterviewState(
            interview_plan_id=interview_plan.plan_id,
            current_phase=InterviewPhase.OPENING,
            time_remaining_minutes=interview_plan.total_estimated_duration_minutes
        )
        
        # Question management
        self.current_section_index = 0
        self.current_question_index = 0
        self.questions_asked = []
        self.follow_up_queue = []
        
        # Response evaluation
        self.response_evaluations = []
        self.performance_trend = "neutral"
        
        # Initialize context service for interview-specific context
        self.interview_context_service = ContextInjectionService()
        
        # Initialize follow-up configuration
        self.followup_config = FollowUpConfig()
        
        self.logger.info(f"Interview Conductor Agent initialized with plan: {interview_plan.plan_id}")
        self.logger.info(f"Interview sections: {len(interview_plan.interview_sections)}")
        self.logger.info(f"Total estimated duration: {interview_plan.total_estimated_duration_minutes} minutes")
        self.logger.info(f"Follow-up limits: {self.followup_config.max_followups_per_question} per question, {self.followup_config.max_followups_per_section} per section")
    
    def _generate_interview_instructions(self, interview_plan: InterviewPlan) -> str:
        """Generate specialized system instructions for interview conducting."""
        print("InterviewConductorAgent._generate_interview_instructions is called")
        focus_areas = interview_plan.key_focus_areas
        evaluation_priorities = interview_plan.evaluation_priorities
        
        instructions = f"""You are a Professional Interview Conductor AI. You are conducting a structured interview based on a detailed interview plan.

CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

INTERVIEW OVERVIEW:
- Total Sections: {len(interview_plan.interview_sections)}
- Estimated Duration: {interview_plan.total_estimated_duration_minutes} minutes
- Key Focus Areas: {', '.join(focus_areas) if focus_areas else 'General assessment'}
- Evaluation Priorities: {', '.join(evaluation_priorities) if evaluation_priorities else 'Overall competency'}

YOUR ROLE AND RESPONSIBILITIES:
1. **Professional Interviewer**: Maintain a warm, professional, and engaging tone throughout
2. **Context-Aware**: Use the provided interview plan and candidate information to ask relevant questions
3. **Dynamic Questioner**: Adapt questions based on candidate responses and background
4. **Time Manager**: Keep track of interview progression and manage time effectively
5. **Active Listener**: Ask thoughtful follow-up questions based on responses
6. **English Language**: ALWAYS respond in English, regardless of what language the candidate uses

INTERVIEW CONDUCTING GUIDELINES:

**Question Selection:**
- Ask questions appropriate for the current interview phase
- Choose questions that align with the candidate's background and the job requirements
- Prioritize questions that will reveal key competencies for the role
- Ask follow-up questions to clarify or dive deeper into responses

**Interview Flow Management:**
- Follow the structured interview plan but remain flexible
- Smoothly transition between interview sections
- Adjust pacing based on candidate responses and time constraints
- Guide the conversation back on track if it goes off-topic

**Response Evaluation:**
- Listen carefully to candidate responses
- Identify strengths, skills, and potential areas of concern
- Ask probing questions to validate claims or explore gaps
- Take note of communication style, problem-solving approach, and cultural fit

**Professional Communication:**
- Use clear, concise language appropriate for the role level
- Provide context for questions when helpful
- Show genuine interest in the candidate's experiences
- Maintain professionalism while being conversational

**Time and Pace Management:**
- Monitor interview progression against the planned schedule
- Allocate appropriate time to critical assessment areas
- Provide smooth transitions between sections
- Wrap up sections naturally when objectives are met

CURRENT INTERVIEW CONTEXT:
You will receive detailed context about:
- The specific interview plan and current section
- Candidate's resume and background information  
- Job requirements and matching analysis
- Previous responses and evaluation notes
- Recommended questions and follow-up areas

FOLLOW-UP QUESTION GUIDELINES:
- You have a STRICT MAXIMUM of 2 follow-up questions per original question
- Current follow-up count will be shown as [Follow-up X/2] in the user's message
- When you see [Follow-up 2/2], you MUST move to the next question after this response
- Start follow-ups with: "Let me ask a follow-up question about that..."
- Ask follow-ups ONLY when responses lack critical detail, examples, or clarity
- Follow-up examples: "Let me ask a follow-up question about that. Can you give me a specific example?", "Let me ask a follow-up question about that. What was the outcome?"
- If the response is adequate (even if not perfect), move to the next question
- When moving to next question, you can acknowledge previous answers: "Thank you for sharing about [topic]. Now let's move on to..."

RESPONSE FORMAT:
- Ask ONE clear question at a time
- Provide brief context if needed ("Now let's talk about your technical experience...")
- Keep questions conversational but purposeful
- Avoid yes/no questions when possible - encourage detailed responses
- When asking follow-ups, reference the previous response: "You mentioned X, can you elaborate on..."

DECISION MAKING:
- Move to next question if response quality is excellent (detailed, relevant, with examples)
- Ask follow-up if response is brief, vague, or lacks examples
- Consider time spent on current question vs. interview progress
- Balance thoroughness with interview completion

LANGUAGE REQUIREMENT:
- ALWAYS respond in English, even if the candidate speaks in another language
- If the candidate uses a different language, politely acknowledge it but continue the interview in English
- Example: "I understand, but let's continue our interview in English. Now, regarding your experience..."

Remember: Your goal is to conduct a thorough, fair, and engaging interview that accurately assesses the candidate's fit for the role while providing a positive candidate experience.
"""
        
        return instructions
    
    async def conduct_interview_turn(self, 
                                   candidate_response: str,
                                   additional_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Conduct one turn of the interview based on candidate response.
        
        This method uses proper structured context injection to provide the agent
        with complete interview context instead of flattened text.
        
        Args:
            candidate_response: What the candidate just said
            additional_context: Additional context data
            
        Returns:
            AgentResponse with next question or comment
        """
        print("InterviewConductorAgent.conduct_interview_turn is called")
        try:
            # Update interview state
            self._update_interview_state(candidate_response)
            
            # Prepare structured interview context 
            interview_context = self._prepare_interview_context(candidate_response, additional_context)
            
            # Process with structured context (no more text flattening!)
            # response = await self.process_request(
            #     candidate_response, 
            #     interview_context
            # )

            # Add explicit follow-up guidance if at limit
            if self.interview_state.current_question_followup_count >= 2:
                candidate_response = f"[SYSTEM NOTE: You have reached the maximum of 2 follow-ups. Please move to the next question.]\n\n{candidate_response}"
            
            # Process with structured context (no more text flattening!)
            response = await self.process_request(
                candidate_response, 
                interview_context
            )
            
            # Update question tracking
            if response.success:
                self._track_question_asked(response.content)
                self.logger.info(f"Interview turn completed - Phase: {self.interview_state.current_phase.value}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in interview turn: {e}")
            return AgentResponse(
                success=False,
                content="I apologize, let me ask you another question.",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                error=str(e)
            )
    
    def _prepare_interview_context(self, 
                                 candidate_response: str,
                                 additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare structured interview context for the agent.
        
        This is the CRITICAL FIX - instead of flattening to text, we provide
        structured context that the enhanced base agent can properly handle.
        """
        print("InterviewConductorAgent._prepare_interview_context is called")
        # Get current section information
        current_section = self._get_current_section()
        
        # Build comprehensive interview context
        interview_context = {
            "interview_plan": self.interview_plan,
            "interview_state": self.interview_state,
            "current_section": {
                "section_name": current_section.section_name if current_section else "Unknown",
                "phase": self.interview_state.current_phase.value,
                "objectives": current_section.objectives if current_section else [],
                "key_evaluation_points": current_section.key_evaluation_points if current_section else []
            },
            "response_history": [
                {
                    "question_asked": q,
                    "timestamp": datetime.now().isoformat()
                } 
                for q in self.questions_asked[-3:]  # Last 3 questions for context
            ],
            "interview_progress": {
                "current_section_index": self.current_section_index,
                "total_sections": len(self.interview_plan.interview_sections),
                "questions_asked_count": len(self.questions_asked),
                "time_remaining": self.interview_state.time_remaining_minutes,
                "performance_trend": self.performance_trend
            },
            # "followup_context": {
            #     "current_followup_count": self.interview_state.current_question_followup_count,
            #     "max_followups_allowed": self.followup_config.max_followups_per_question,
            #     "response_quality": self.interview_state.current_question_response_quality,
            #     "should_ask_followup": self.interview_state.current_question_followup_count > 0,
            #     "time_spent_on_question": self._get_time_spent_on_current_question(),
            #     "followup_config": {
            #         "min_quality_threshold": self.followup_config.minimum_response_quality,
            #         "excellent_threshold": self.followup_config.excellent_response_threshold,
            #         "max_time_per_question": self.followup_config.max_time_per_question_minutes
            #     }
            # },
            "followup_context": {
                "current_followup_count": self.interview_state.current_question_followup_count,
                "max_followups_allowed": 2,  # Hardcode to 2 for clarity
                "response_quality": self.interview_state.current_question_response_quality,
                "should_ask_followup": self._should_ask_followup(candidate_response, self.interview_state.current_question_response_quality),
                "must_move_to_next": self.interview_state.current_question_followup_count >= 2,  # Explicit flag
                "time_spent_on_question": self._get_time_spent_on_current_question(),
                "followup_config": {
                    "min_quality_threshold": self.followup_config.minimum_response_quality,
                    "excellent_threshold": self.followup_config.excellent_response_threshold,
                    "max_time_per_question": self.followup_config.max_time_per_question_minutes
                }
            },
            "candidate_background": self._extract_candidate_background(),
            "job_requirements": self._extract_job_requirements(),
            "evaluation_notes": [eval.evaluator_notes for eval in self.response_evaluations if eval.evaluator_notes],
            "follow_up_suggestions": self.follow_up_queue.copy(),
            "next_recommended_questions": self._get_recommended_questions()
        }
        
        # Add any additional context
        if additional_context:
            interview_context.update(additional_context)
        
        return interview_context
    
    def _get_current_section(self):
        """Get current interview section."""
        
        if 0 <= self.current_section_index < len(self.interview_plan.interview_sections):
            return self.interview_plan.interview_sections[self.current_section_index]
        return None
    
    def _extract_candidate_background(self) -> Dict[str, Any]:
        """Extract relevant candidate background from interview plan."""
        if not self.interview_plan.resume_analysis:
            return {}
        
        resume = self.interview_plan.resume_analysis
        return {
            "technical_skills": [skill.skill_name for skill in resume.technical_skills],
            "work_experience": [
                {
                    "title": exp.job_title,
                    "company": exp.company_name,
                    "duration": exp.duration
                }
                for exp in resume.work_experience
            ],
            "strengths": resume.strengths,
            "potential_gaps": resume.potential_gaps
        }
    
    def _extract_job_requirements(self) -> Dict[str, Any]:
        """Extract key job requirements from interview plan."""
        if not self.interview_plan.job_analysis:
            return {}
        
        job = self.interview_plan.job_analysis
        return {
            "job_title": job.job_title,
            "key_technical_requirements": [req.requirement for req in job.technical_requirements],
            "experience_requirements": [req.requirement for req in job.experience_requirements],
            "critical_success_factors": job.critical_success_factors
        }
    
    def _get_recommended_questions(self) -> List[Dict[str, Any]]:
        """Get recommended questions for current section."""
        current_section = self._get_current_section()
        if not current_section:
            return []
        
        return [
            {
                "question_text": q.question_text,
                "question_type": q.question_type.value,
                "difficulty_level": q.difficulty_level,
                "skill_focus": q.skill_focus
            }
            for q in current_section.questions[self.current_question_index:self.current_question_index+3]
        ]
    
    def _update_interview_state(self, candidate_response: str):
        """Enhanced interview state update with follow-up decision logic."""
        print("InterviewConductorAgent._update_interview_state is called")
        
        # Evaluate response quality using enhanced evaluation
        response_quality = self._evaluate_response_quality(candidate_response)
        self.interview_state.current_question_response_quality = response_quality
        
        # Create detailed evaluation
        evaluation = ResponseEvaluation(
            response_text=candidate_response,
            question_id=f"q_{len(self.questions_asked)}_{self.interview_state.current_question_followup_count}",
            overall_response_score=response_quality,
            strengths_demonstrated=self._identify_response_strengths(candidate_response),
            evaluator_notes=f"Response quality: {response_quality:.2f}, Follow-ups: {self.interview_state.current_question_followup_count}"
        )
        
        self.response_evaluations.append(evaluation)
        self.interview_state.responses_received.append(evaluation)
        
        # Check if this is a follow-up or new question
        if self.interview_state.current_question_followup_count == 0:
            # This is the first response to a new question
            self.interview_state.current_question_start_time = datetime.now()
            self.interview_state.questions_asked.append(f"New question at {datetime.now().isoformat()}")
        else:
            # This is a follow-up response
            self.interview_state.total_followups_asked += 1
        
        # Determine next action: follow-up or move to next question
        should_ask_followup = self._should_ask_followup(candidate_response, response_quality)
        
        if should_ask_followup:
            # Increment follow-up counter but don't advance question
            self.interview_state.current_question_followup_count += 1
            self.logger.info(f"Planning follow-up question #{self.interview_state.current_question_followup_count}")
        else:
            # Move to next question
            self._advance_to_next_question()
        
        # Update time tracking
        self._update_time_tracking()
    
    def _evaluate_response_quality(self, response: str) -> float:
        """Evaluate response quality on a 0-1 scale using multiple criteria."""
        
        # Basic metrics
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        # Completeness score (0-1)
        completeness = min(1.0, word_count / 50)  # Assume 50 words is complete
        if word_count < 10:
            completeness = 0.2
        
        # Relevance score (simplified - could use AI for better evaluation)
        relevance = 0.8 if word_count > 15 else 0.5  # Placeholder logic
        
        # Depth score (based on examples, details, reasoning)
        depth_indicators = ['because', 'example', 'specifically', 'when', 'how', 'why', 'result', 'impact']
        depth_count = sum(1 for indicator in depth_indicators if indicator.lower() in response.lower())
        depth = min(1.0, depth_count / 3)  # Normalize to 0-1
        
        # Clarity score (based on structure and coherence)
        clarity = 0.9 if sentence_count > 1 and word_count > 20 else 0.6
        
        # Calculate weighted score
        config = self.followup_config
        quality_score = (
            completeness * config.completeness_weight +
            relevance * config.relevance_weight +
            depth * config.depth_weight +
            clarity * config.clarity_weight
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def _identify_response_strengths(self, response: str) -> List[str]:
        """Identify strengths demonstrated in the response."""
        strengths = []
        
        if len(response.split()) > 30:
            strengths.append("Detailed response")
        
        if any(word in response.lower() for word in ['example', 'specifically', 'instance']):
            strengths.append("Provided examples")
        
        if any(word in response.lower() for word in ['result', 'outcome', 'impact', 'achieved']):
            strengths.append("Outcome-focused")
        
        if any(word in response.lower() for word in ['team', 'collaborate', 'together']):
            strengths.append("Team collaboration")
        
        if any(word in response.lower() for word in ['challenge', 'problem', 'difficult']):
            strengths.append("Problem-solving")
        
        return strengths if strengths else ["Provided response"]
    
    def _should_ask_followup(self, response: str, quality_score: float) -> bool:
        """Determine if a follow-up question should be asked."""
        
        # Get current question info
        current_section = self._get_current_section()
        if not current_section:
            return False
        
        current_question = None
        if self.current_question_index < len(current_section.questions):
            current_question = current_section.questions[self.current_question_index]
        
        # Check follow-up limits
        max_followups = self.followup_config.max_followups_per_question
        if current_question:
            max_followups = self.followup_config.get_followup_limit_for_question_type(
                current_question.question_type.value
            )
        
        # Reason 1: Reached follow-up limit
        if self.interview_state.current_question_followup_count >= max_followups:
            self.logger.info(f"Reached follow-up limit ({max_followups})")
            return False
        
        # Reason 2: Response quality is excellent - no need for follow-up
        if quality_score >= self.followup_config.excellent_response_threshold:
            self.logger.info(f"Excellent response quality ({quality_score:.2f}) - skipping follow-up")
            return False
        
        # Reason 3: Time constraints
        if self._is_time_constrained():
            self.logger.info("Time constrained - moving to next question")
            return False
        
        # Reason 4: Response quality is below minimum - needs follow-up
        if quality_score < self.followup_config.minimum_response_quality:
            self.logger.info(f"Low response quality ({quality_score:.2f}) - asking follow-up")
            return True
        
        # Reason 5: Response is adequate but could be improved
        if quality_score < 0.75 and self.interview_state.current_question_followup_count == 0:
            self.logger.info(f"Adequate response ({quality_score:.2f}) - one follow-up")
            return True
        
        # Default: move to next question
        return False
    
    def _is_time_constrained(self) -> bool:
        """Check if we're running out of time for the current question or section."""
        
        if not self.interview_state.current_question_start_time:
            return False
        
        # Check time spent on current question
        time_spent = (datetime.now() - self.interview_state.current_question_start_time).total_seconds() / 60
        if time_spent >= self.followup_config.max_time_per_question_minutes:
            return True
        
        # Check remaining section time
        section_time_remaining = self.interview_state.time_remaining_minutes
        if section_time_remaining <= self.followup_config.section_time_buffer_minutes:
            return True
        
        return False
    
    def _advance_to_next_question(self):
        """Advance to the next question and reset follow-up tracking."""
        
        # Reset follow-up tracking for new question
        self.interview_state.current_question_followup_count = 0
        self.interview_state.current_question_start_time = None
        self.interview_state.current_question_response_quality = 0.0
        
        # Advance question index
        self.current_question_index += 1
        
        # Check if should move to next section
        current_section = self._get_current_section()
        if current_section and self.current_question_index >= len(current_section.questions):
            self._advance_to_next_section()
        
        self.logger.info(f"Advanced to question {self.current_question_index} in section {self.current_section_index}")
    
    def _update_time_tracking(self):
        """Update time tracking for the interview."""
        
        # Simple time update (could be more sophisticated)
        if self.interview_state.current_question_followup_count == 0:
            # New question - estimate 2 minutes
            time_used = 2
        else:
            # Follow-up - estimate 1 minute
            time_used = 1
        
        self.interview_state.time_remaining_minutes = max(0, 
            self.interview_state.time_remaining_minutes - time_used
        )
    
    def _get_time_spent_on_current_question(self) -> float:
        """Get time spent on current question in minutes."""
        if not self.interview_state.current_question_start_time:
            return 0.0
        
        time_spent = (datetime.now() - self.interview_state.current_question_start_time).total_seconds() / 60
        return round(time_spent, 2)
    
    def _advance_to_next_section(self):
        """Advance to the next interview section."""
        self.current_section_index += 1
        self.current_question_index = 0
        
        if self.current_section_index < len(self.interview_plan.interview_sections):
            next_section = self.interview_plan.interview_sections[self.current_section_index]
            self.interview_state.current_phase = next_section.phase
            self.logger.info(f"Advanced to section: {next_section.section_name}")
        else:
            self.interview_state.current_phase = InterviewPhase.CLOSING
            self.interview_state.should_continue = False
            self.logger.info("Interview completed - all sections covered")
    
    def _track_question_asked(self, question: str):
        """Track questions asked during interview."""
        print("InterviewConductorAgent._track_question_asked is called")
        self.questions_asked.append(question)
        self.interview_state.questions_asked.append(question)
    
    def get_interview_progress(self) -> Dict[str, Any]:
        """Get current interview progress and state."""
        print("InterviewConductorAgent.get_interview_progress is called. Here are the ")
        current_section = self._get_current_section()
        
        return {
            "interview_id": self.interview_plan.plan_id,
            "current_phase": self.interview_state.current_phase.value,
            "section_progress": {
                "current_section": self.current_section_index + 1,
                "total_sections": len(self.interview_plan.interview_sections),
                "current_section_name": current_section.section_name if current_section else "Completed"
            },
            "question_progress": {
                "questions_asked": len(self.questions_asked),
                "current_question_index": self.current_question_index
            },
            "time_management": {
                "time_remaining_minutes": self.interview_state.time_remaining_minutes,
                "estimated_duration": self.interview_plan.total_estimated_duration_minutes
            },
            "performance_indicators": {
                "responses_evaluated": len(self.response_evaluations),
                "performance_trend": self.performance_trend
            },
            "interview_status": {
                "should_continue": self.interview_state.should_continue,
                "completion_percentage": self._calculate_completion_percentage()
            }
        }
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate interview completion percentage."""
        print("InterviewConductorAgent._calculate_completion_percentage is called")
        if not self.interview_plan.interview_sections:
            return 100.0
        
        section_progress = (self.current_section_index / len(self.interview_plan.interview_sections)) * 100
        return min(100.0, section_progress)
    
    def generate_interview_summary(self) -> Dict[str, Any]:
        """Generate comprehensive interview summary."""
        print("InterviewConductorAgent.generate_interview_summary is called")
        return {
            "interview_metadata": {
                "interview_id": self.interview_plan.plan_id,
                "duration_minutes": self.interview_plan.total_estimated_duration_minutes - self.interview_state.time_remaining_minutes,
                "questions_asked": len(self.questions_asked),
                "sections_completed": self.current_section_index,
                "completion_status": "completed" if not self.interview_state.should_continue else "in_progress"
            },
            "performance_summary": {
                "total_responses": len(self.response_evaluations),
                "average_response_quality": self._calculate_average_response_quality(),
                "performance_trend": self.performance_trend,
                "strengths_identified": self._aggregate_strengths(),
                "areas_for_improvement": self._aggregate_improvement_areas()
            },
            "interview_sections": [
                {
                    "section_name": section.section_name,
                    "phase": section.phase.value,
                    "completed": i < self.current_section_index,
                    "questions_from_section": len([q for q in self.questions_asked if i < self.current_section_index])
                }
                for i, section in enumerate(self.interview_plan.interview_sections)
            ],
            "recommendations": {
                "overall_assessment": self._generate_overall_assessment(),
                "follow_up_areas": self.follow_up_queue,
                "interviewer_notes": self.interview_state.interviewer_notes
            }
        }
    
    def _calculate_average_response_quality(self) -> float:
        """Calculate average response quality score."""
        print("InterviewConductorAgent._calculate_average_response_quality is called")
        if not self.response_evaluations:
            return 0.0
        
        scores = [eval.overall_response_score for eval in self.response_evaluations if eval.overall_response_score]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _aggregate_strengths(self) -> List[str]:
        """Aggregate strengths identified during interview."""
        print("InterviewConductorAgent._aggregate_strengths is called")
        strengths = []
        for eval in self.response_evaluations:
            strengths.extend(eval.strengths_demonstrated)
        return list(set(strengths))  # Remove duplicates
    
    def _aggregate_improvement_areas(self) -> List[str]:
        """Aggregate areas for improvement identified during interview."""
        print("InterviewConductorAgent._aggregate_improvement_areas is called")
        improvements = []
        for eval in self.response_evaluations:
            improvements.extend(eval.weaknesses_identified)
        return list(set(improvements))
    
    def _generate_overall_assessment(self) -> str:
        """Generate overall assessment based on interview performance."""
        print("InterviewConductorAgent._generate_overall_assessment is called")
        completion_pct = self._calculate_completion_percentage()
        avg_quality = self._calculate_average_response_quality()
        
        if completion_pct >= 90 and avg_quality >= 3.5:
            return "Strong candidate - completed interview with high-quality responses"
        elif completion_pct >= 75 and avg_quality >= 3.0:
            return "Good candidate - solid interview performance with some strong areas"
        elif completion_pct >= 50:
            return "Mixed results - interview partially completed with varying response quality"
        else:
            return "Limited assessment - insufficient interview completion for full evaluation"


# Convenience functions for creating interview conductor agents
def create_interview_conductor(interview_plan: InterviewPlan,
                             agent_name: str = "Interview Conductor") -> InterviewConductorAgent:
    """Create an interview conductor agent with the given interview plan."""
    print("InterviewConductorAgent.create_interview_conductor is called")
    return InterviewConductorAgent(interview_plan, agent_name)


async def test_interview_conductor():
    """Test the interview conductor agent functionality."""
    print("Testing Interview Conductor Agent...")
    print("=" * 60)
    
    # This would require a proper interview plan - placeholder for now
    from models.interview_models import InterviewPlan, InterviewSection, Question, InterviewPhase, QuestionType
    
    # Create sample interview plan
    sample_questions = [
        Question(
            question_text="Tell me about your background and what interests you about this role.",
            question_type=QuestionType.OPEN_ENDED,
            phase=InterviewPhase.OPENING,
            difficulty_level=1,
            estimated_time_minutes=3
        ),
        Question(
            question_text="Describe a challenging technical problem you've solved recently.",
            question_type=QuestionType.TECHNICAL,
            phase=InterviewPhase.TECHNICAL,
            difficulty_level=3,
            estimated_time_minutes=5
        )
    ]
    
    sample_section = InterviewSection(
        phase=InterviewPhase.OPENING,
        section_name="Opening Questions",
        description="Initial conversation and background",
        objectives=["Get candidate comfortable", "Understand motivation"],
        estimated_duration_minutes=10,
        questions=sample_questions
    )
    
    sample_plan = InterviewPlan(
        interview_sections=[sample_section],
        total_estimated_duration_minutes=30,
        interview_objectives=["Assess technical skills", "Evaluate cultural fit"],
        key_focus_areas=["Python programming", "Problem solving"]
    )
    
    try:
        # Create interview conductor
        conductor = InterviewConductorAgent(sample_plan, "Test Interview Conductor")
        
        print(f"✅ Created interview conductor for plan: {sample_plan.plan_id}")
        
        # Simulate interview turn
        response = await conductor.conduct_interview_turn(
            "Hello, I'm excited about this opportunity. I have 5 years of Python experience and I'm passionate about building scalable systems."
        )
        
        if response.success:
            print(f"✅ Interview turn successful")
            print(f"   Agent response: {response.content[:100]}...")
            print(f"   Processing time: {response.processing_time:.2f}s")
        else:
            print(f"❌ Interview turn failed: {response.error}")
        
        # Get interview progress
        progress = conductor.get_interview_progress()
        print(f"\n📊 Interview Progress:")
        print(f"   Current phase: {progress['current_phase']}")
        print(f"   Questions asked: {progress['question_progress']['questions_asked']}")
        print(f"   Completion: {progress['interview_status']['completion_percentage']:.1f}%")
        
        # Generate summary
        summary = conductor.generate_interview_summary()
        print(f"\n📋 Interview Summary:")
        print(f"   Status: {summary['interview_metadata']['completion_status']}")
        print(f"   Responses evaluated: {summary['performance_summary']['total_responses']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_interview_conductor())
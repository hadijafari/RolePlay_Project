"""
Interview Data Models
Pydantic models for structured interview data and context management.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class SkillLevel(str, Enum):
    """Skill proficiency levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class InterviewPhase(str, Enum):
    """Interview phases for structured progression."""
    OPENING = "opening"
    TECHNICAL = "technical"
    EXPERIENCE = "experience"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"
    CLOSING = "closing"


class QuestionType(str, Enum):
    """Types of interview questions."""
    OPEN_ENDED = "open_ended"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"


class DocumentType(str, Enum):
    """Supported document types."""
    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"
    COVER_LETTER = "cover_letter"


# ===== RESUME ANALYSIS MODELS =====

class PersonalInfo(BaseModel):
    """Personal information from resume."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class TechnicalSkill(BaseModel):
    """Technical skill with proficiency assessment."""
    skill_name: str
    category: str  # e.g., "Programming Languages", "Frameworks", "Tools"
    proficiency_level: Optional[SkillLevel] = None
    years_experience: Optional[int] = None
    mentioned_count: int = 0
    context_mentions: List[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    """Work experience entry."""
    job_title: str
    company_name: str
    duration: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    key_responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    technologies_used: List[str] = Field(default_factory=list)
    team_size: Optional[int] = None
    reporting_structure: Optional[str] = None


class Education(BaseModel):
    """Educational background."""
    degree: str
    major: Optional[str] = None
    institution: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None
    relevant_coursework: List[str] = Field(default_factory=list)
    honors: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    """Professional certification."""
    name: str
    issuing_organization: str
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    is_active: bool = True


class CareerProgression(BaseModel):
    """Career progression analysis."""
    total_experience_years: float
    career_trajectory: str  # "ascending", "lateral", "mixed"
    job_changes_frequency: float  # years between job changes
    industry_consistency: bool
    role_progression: List[str]  # progression path
    career_gaps: List[str] = Field(default_factory=list)


class ResumeAnalysis(BaseModel):
    """Complete resume analysis structure."""
    document_type: DocumentType = DocumentType.RESUME
    personal_info: PersonalInfo
    technical_skills: List[TechnicalSkill] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    career_progression: Optional[CareerProgression] = None
    
    # Analysis insights
    strengths: List[str] = Field(default_factory=list)
    potential_gaps: List[str] = Field(default_factory=list)
    unique_selling_points: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Confidence score must be between 0 and 1')
        return v


# ===== JOB DESCRIPTION ANALYSIS MODELS =====

class JobRequirement(BaseModel):
    """Individual job requirement."""
    requirement: str
    category: str  # "technical", "soft_skill", "experience", "education"
    priority: str  # "required", "preferred", "nice_to_have"
    years_required: Optional[int] = None


class CompanyInfo(BaseModel):
    """Company information from job description."""
    company_name: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    culture_keywords: List[str] = Field(default_factory=list)
    benefits_offered: List[str] = Field(default_factory=list)


class JobDescriptionAnalysis(BaseModel):
    """Complete job description analysis structure."""
    document_type: DocumentType = DocumentType.JOB_DESCRIPTION
    job_title: str
    company_info: CompanyInfo
    
    # Core requirements
    technical_requirements: List[JobRequirement] = Field(default_factory=list)
    soft_skill_requirements: List[JobRequirement] = Field(default_factory=list)
    experience_requirements: List[JobRequirement] = Field(default_factory=list)
    education_requirements: List[JobRequirement] = Field(default_factory=list)
    
    # Job details
    job_summary: str
    key_responsibilities: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)
    career_growth_opportunities: List[str] = Field(default_factory=list)
    
    # Analysis insights
    role_seniority: str  # "junior", "mid", "senior", "lead", "executive"
    role_type: str  # "individual_contributor", "team_lead", "manager", "director"
    critical_success_factors: List[str] = Field(default_factory=list)
    potential_challenges: List[str] = Field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None


# ===== INTERVIEW PLANNING MODELS =====

class Question(BaseModel):
    """Interview question with metadata."""
    question_text: str
    question_type: QuestionType
    phase: InterviewPhase
    difficulty_level: int = Field(ge=1, le=5)  # 1=easy, 5=hard
    estimated_time_minutes: int = Field(ge=1, le=15)
    
    # Question context
    skill_focus: List[str] = Field(default_factory=list)
    evaluation_criteria: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    
    # Scoring guidance
    excellent_response_indicators: List[str] = Field(default_factory=list)
    red_flag_indicators: List[str] = Field(default_factory=list)


class InterviewSection(BaseModel):
    """Interview section with questions and objectives."""
    phase: InterviewPhase
    section_name: str
    description: str
    objectives: List[str] = Field(default_factory=list)
    estimated_duration_minutes: int = Field(ge=5, le=60)
    
    questions: List[Question] = Field(default_factory=list)
    backup_questions: List[Question] = Field(default_factory=list)
    
    # Section-specific evaluation
    key_evaluation_points: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class CandidateMatch(BaseModel):
    """Candidate-job matching analysis."""
    overall_match_score: float = Field(ge=0, le=1)
    
    # Detailed matching
    technical_skills_match: Dict[str, float] = Field(default_factory=dict)
    experience_match_score: float = Field(ge=0, le=1)
    education_match_score: float = Field(ge=0, le=1)
    
    # Match analysis
    strong_matches: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    growth_potential_areas: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    
    # Interview focus recommendations
    areas_to_probe: List[str] = Field(default_factory=list)
    strengths_to_validate: List[str] = Field(default_factory=list)
    risks_to_assess: List[str] = Field(default_factory=list)


class InterviewPlan(BaseModel):
    """Complete interview plan with all sections and evaluation criteria."""
    # Plan metadata
    plan_id: str = Field(default_factory=lambda: f"interview_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Source data references
    resume_analysis: Optional[ResumeAnalysis] = None
    job_analysis: Optional[JobDescriptionAnalysis] = None
    candidate_match: Optional[CandidateMatch] = None
    
    # Interview structure
    interview_sections: List[InterviewSection] = Field(default_factory=list)
    total_estimated_duration_minutes: int = 30
    
    # Interview strategy
    interview_objectives: List[str] = Field(default_factory=list)
    key_focus_areas: List[str] = Field(default_factory=list)
    evaluation_priorities: List[str] = Field(default_factory=list)
    
    # Risk management
    potential_red_flags: List[str] = Field(default_factory=list)
    clarification_needed: List[str] = Field(default_factory=list)
    
    # Interviewer guidance
    interviewer_notes: List[str] = Field(default_factory=list)
    recommended_follow_up_areas: List[str] = Field(default_factory=list)
    
    @validator('total_estimated_duration_minutes')
    def calculate_total_duration(cls, v, values):
        if 'interview_sections' in values:
            calculated_duration = sum(section.estimated_duration_minutes 
                                    for section in values['interview_sections'])
            return calculated_duration
        return v


# ===== INTERVIEW EXECUTION MODELS =====

class ResponseEvaluation(BaseModel):
    """Evaluation of a candidate's response."""
    response_text: str
    question_id: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Scoring
    technical_accuracy_score: Optional[float] = Field(None, ge=0, le=5)
    communication_clarity_score: Optional[float] = Field(None, ge=0, le=5)
    depth_of_knowledge_score: Optional[float] = Field(None, ge=0, le=5)
    problem_solving_score: Optional[float] = Field(None, ge=0, le=5)
    overall_response_score: Optional[float] = Field(None, ge=0, le=5)
    
    # Qualitative assessment
    strengths_demonstrated: List[str] = Field(default_factory=list)
    weaknesses_identified: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    follow_up_needed: List[str] = Field(default_factory=list)
    
    # Interviewer notes
    evaluator_notes: Optional[str] = None
    confidence_in_evaluation: Optional[float] = Field(None, ge=0, le=1)


class InterviewState(BaseModel):
    """Current state of interview execution."""
    interview_plan_id: str
    current_section_index: int = 0
    current_question_index: int = 0
    
    # Progress tracking
    started_timestamp: datetime = Field(default_factory=datetime.now)
    current_phase: InterviewPhase = InterviewPhase.OPENING
    completed_sections: List[str] = Field(default_factory=list)
    
    # Dynamic adjustments
    time_remaining_minutes: int = 60
    questions_asked: List[str] = Field(default_factory=list)
    responses_received: List[ResponseEvaluation] = Field(default_factory=list)
    
    # Follow-up question tracking
    current_question_followup_count: int = 0
    current_question_start_time: Optional[datetime] = None
    total_followups_asked: int = 0
    current_question_response_quality: float = 0.0  # 0-1 scale
    
    # Interview flow control
    should_continue: bool = True
    early_termination_reason: Optional[str] = None
    interviewer_notes: List[str] = Field(default_factory=list)
    
    # Real-time insights
    current_performance_trend: Optional[str] = None
    areas_needing_more_time: List[str] = Field(default_factory=list)
    sections_to_skip: List[str] = Field(default_factory=list)


# ===== CONTEXT MODELS FOR AGENTS =====

class InterviewContext(BaseModel):
    """Structured context for interview conductor agent."""
    interview_plan: InterviewPlan
    interview_state: InterviewState
    candidate_responses_history: List[ResponseEvaluation] = Field(default_factory=list)
    
    # Dynamic context
    current_objective: str
    next_recommended_question: Optional[Question] = None
    follow_up_suggestions: List[str] = Field(default_factory=list)
    
    # Evaluation context
    performance_summary: Optional[Dict[str, Any]] = None
    real_time_adjustments: List[str] = Field(default_factory=list)


class DocumentAnalysisContext(BaseModel):
    """Context for document analysis operations."""
    document_type: DocumentType
    file_path: str
    extraction_method: str  # "text", "vision", "hybrid"
    
    # Processing metadata
    file_size_bytes: int
    processing_start_time: datetime = Field(default_factory=datetime.now)
    confidence_threshold: float = 0.7
    
    # Analysis preferences
    focus_areas: List[str] = Field(default_factory=list)
    skip_sections: List[str] = Field(default_factory=list)


# ===== UTILITY MODELS =====

class ProcessingResult(BaseModel):
    """Generic result structure for processing operations."""
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# Export all models
__all__ = [
    # Enums
    'SkillLevel', 'InterviewPhase', 'QuestionType', 'DocumentType',
    
    # Resume Models
    'PersonalInfo', 'TechnicalSkill', 'WorkExperience', 'Education', 
    'Certification', 'CareerProgression', 'ResumeAnalysis',
    
    # Job Description Models
    'JobRequirement', 'CompanyInfo', 'JobDescriptionAnalysis',
    
    # Interview Planning Models
    'Question', 'InterviewSection', 'CandidateMatch', 'InterviewPlan',
    
    # Interview Execution Models
    'ResponseEvaluation', 'InterviewState',
    
    # Context Models
    'InterviewContext', 'DocumentAnalysisContext',
    
    # Utility Models
    'ProcessingResult',
    
    # Configuration Models
    'FollowUpConfig'
]


# ===== FOLLOW-UP CONFIGURATION =====

class FollowUpConfig(BaseModel):
    """Configuration for follow-up question management."""
    
    # Follow-up limits
    max_followups_per_question: int = 2  # Maximum follow-ups per original question
    max_followups_per_section: int = 8   # Maximum follow-ups per interview section
    max_time_per_question_minutes: float = 8.0  # Maximum time to spend on one question
    
    # Response quality thresholds (0-1 scale)
    minimum_response_quality: float = 0.6  # Minimum quality to move to next question
    excellent_response_threshold: float = 0.8  # Skip follow-ups if response is excellent
    
    # Time management
    followup_time_limit_minutes: float = 3.0  # Maximum time for follow-up responses
    section_time_buffer_minutes: float = 5.0  # Buffer time to ensure section completion
    
    # Evaluation criteria weights
    completeness_weight: float = 0.4  # How complete is the response?
    relevance_weight: float = 0.3     # How relevant to the question?
    depth_weight: float = 0.2         # How detailed/insightful?
    clarity_weight: float = 0.1       # How clear is communication?
    
    # Question type specific settings
    technical_question_followup_limit: int = 3  # Technical questions may need more follow-ups
    behavioral_question_followup_limit: int = 2 # Behavioral questions need fewer follow-ups
    
    @validator('completeness_weight', 'relevance_weight', 'depth_weight', 'clarity_weight')
    def validate_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v
    
    @validator('minimum_response_quality', 'excellent_response_threshold')
    def validate_quality_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Quality threshold must be between 0 and 1')
        return v
    
    def get_followup_limit_for_question_type(self, question_type: str) -> int:
        """Get follow-up limit based on question type."""
        if question_type.lower() in ['technical', 'coding', 'system_design']:
            return self.technical_question_followup_limit
        elif question_type.lower() in ['behavioral', 'situational']:
            return self.behavioral_question_followup_limit
        else:
            return self.max_followups_per_question
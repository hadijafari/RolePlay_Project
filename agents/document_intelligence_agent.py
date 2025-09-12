"""
Document Intelligence Agent
Specialized agent for advanced document analysis and interview plan generation.
Uses structured AI processing to analyze resumes and job descriptions.
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
    from services.document_intelligence_service import DocumentIntelligenceService, DocumentParsingError
    from models.interview_models import (
        ResumeAnalysis, JobDescriptionAnalysis, CandidateMatch, 
        InterviewPlan, ProcessingResult, DocumentType,
        SimplifiedResumeAnalysis, SimplifiedPersonalInfo, SimplifiedTechnicalSkill,
        SimplifiedWorkExperience, SimplifiedEducation
    )
except ImportError as e:
    print(f"Document Intelligence Agent: Missing required modules: {e}")
    sys.exit(1)


class DocumentIntelligenceAgent(BaseAgent):
    """
    Specialized agent for document analysis and interview planning.
    
    This agent provides:
    1. Advanced resume analysis using AI and structured outputs
    2. Job description analysis with requirement extraction
    3. Candidate-job matching analysis
    4. Interview plan generation based on document analysis
    5. Multi-format document support (PDF, DOCX, TXT)
    """
    
    def __init__(self, 
                 name: str = "Document Intelligence Agent",
                 model: str = "gpt-4o-mini"):
        """
        Initialize document intelligence agent.
        
        Args:
            name: Agent name
            model: OpenAI model to use
        """
        print("document_intelligence_agent.py: Class DocumentIntelligenceAgent.__init__ called: Initialize document intelligence agent.")
        # Generate specialized instructions for document analysis
        instructions = self._generate_document_analysis_instructions()
        
        # Configure agent for document analysis context
        config = AgentConfig(
            name=name,
            instructions=instructions,
            model=model,
            timeout=90.0,  # Longer timeout for document processing
            max_history_length=50,
            log_level="INFO",
            use_structured_context=True,  # Enable structured context
            agent_type="document"  # Set agent type for proper context handling
        )
        
        # Initialize base agent
        super().__init__(config)
        
        # Initialize document intelligence service
        # print("Initialize document intelligence service")
        try:
            self.document_service = DocumentIntelligenceService()
            self.logger.info("Document Intelligence Service initialized successfully")
            print("Document Intelligence Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Document Intelligence Service: {e}")
            self.document_service = None
        
        # Processing state
        self.current_resume_analysis = None
        self.current_job_analysis = None
        self.current_candidate_match = None
        self.processing_history = []
        
        self.logger.info(f"Document Intelligence Agent initialized: {name}")
        print(f"Document Intelligence Agent initialized: {name}")
    
    def _serialize_datetime_objects(self, data):
        """Serialize datetime objects to ISO strings to prevent JSON serialization errors."""
        
        def serialize_item(item):
            if isinstance(item, datetime):
                return item.isoformat()
            elif isinstance(item, dict):
                return {k: serialize_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [serialize_item(i) for i in item]
            else:
                return item
        
        return serialize_item(data)
    
    def _generate_document_analysis_instructions(self) -> str:
        """Generate specialized system instructions for document analysis."""
        
        return """You are a Document Intelligence AI specialized in analyzing resumes and job descriptions for interview preparation.

CRITICAL CHARACTER RESTRICTIONS:
- Use ONLY standard ASCII characters (A-Z, a-z, 0-9, basic punctuation)
- DO NOT use Unicode symbols like checkmarks (✓), bullet points (•), em dashes (—), smart quotes (" "), or any special characters
- Use simple text formatting: use "- " for bullet points, use regular quotes " and apostrophes '
- Keep all text content compatible with basic ASCII encoding

YOUR EXPERTISE AREAS:
1. **Resume Analysis**: Extract structured data including skills, experience, education, and career progression
2. **Job Description Analysis**: Identify requirements, responsibilities, and success criteria
3. **Candidate Matching**: Assess fit between candidate background and job requirements
4. **Interview Planning**: Generate strategic interview plans based on document analysis

ANALYSIS APPROACH:
- Use structured data extraction with high accuracy
- Identify both explicit and implicit information
- Assess skill levels and experience depth
- Recognize career patterns and progression
- Evaluate cultural and technical fit indicators

COMMUNICATION STYLE:
- Provide detailed, actionable insights
- Structure information clearly with categories
- Highlight key strengths and potential gaps
- Offer specific recommendations for interview focus areas
- Use professional language appropriate for HR and hiring managers

DOCUMENT PROCESSING CAPABILITIES:
- PDF documents with complex layouts
- Microsoft Word documents (DOCX)
- Plain text formats
- Multi-page documents with various sections
- Structured and unstructured content

When analyzing documents, focus on:
1. Completeness of information extraction
2. Accuracy of categorization and classification
3. Relevance to interview and hiring decisions
4. Identification of unique selling points and red flags
5. Strategic recommendations for interview approach

Always provide structured, actionable insights that help improve the interview process and candidate assessment."""
    
    async def analyze_resume(self, 
                           file_path: str,
                           additional_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Analyze resume file and provide structured insights.
        
        Args:
            file_path: Path to resume file
            additional_context: Additional context for analysis
            
        Returns:
            AgentResponse with resume analysis results
        """
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent.analyze_resume called: Analyze resume file and provide structured insights.")
        try:
            self.logger.info(f"Starting resume analysis: {Path(file_path).name}")
            
            if not self.document_service:
                return AgentResponse(
                    success=False,
                    content="Document Intelligence Service not available",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0,
                    error="Service initialization failed"
                )
            
            # Perform document analysis
            analysis_result = await self.document_service.analyze_resume(file_path)
            
            if not analysis_result.success:
                return AgentResponse(
                    success=False,
                    content=f"Resume analysis failed: {analysis_result.error_message}",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds or 0.0,
                    error=analysis_result.error_message
                )
            
            # Convert result to ResumeAnalysis object
            # Remove any datetime strings that might cause issues when recreating the object
            clean_data = analysis_result.result_data.copy()
            if 'analysis_timestamp' in clean_data:
                del clean_data['analysis_timestamp']
            
            resume_analysis = ResumeAnalysis(**clean_data)
            self.current_resume_analysis = resume_analysis
            
            # Generate human-readable analysis summary
            analysis_summary = self._generate_resume_summary(resume_analysis)
            # with open("D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\Third_summary_AI_analysis_response.txt", "w", encoding="utf-8") as f:
            #     f.write(analysis_summary)
            # print(f"In analyze_resume method of Document Intelligence Agent: Generated human-readable analysis summary saved to file")
            # Create structured context for agent processing
            # Serialize resume analysis data to handle datetime objects
            resume_data = self._serialize_datetime_objects(resume_analysis.dict())
            
            context_data = {
                "document_type": "resume",
                "file_path": file_path,
                "resume_analysis": resume_data,
                "analysis_metadata": analysis_result.metadata,
                "processing_time": analysis_result.processing_time_seconds
            }
            
            if additional_context:
                context_data.update(additional_context)
            
            # Process with agent for additional insights
            print(f"In analyze_resume method of Document Intelligence Agent: Processing with agent for additional insights and recommendations")
            agent_response = await self.process_request(
                f"Please provide detailed insights and recommendations based on this resume analysis: {analysis_summary[:500]}...",
                context_data
            )

            # with open("D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\Fourth_enhanced_content_AI_analysis_response.txt", "w", encoding="utf-8") as f:
            #     f.write(agent_response.content)
            # print(f"In analyze_resume method of Document Intelligence Agent: Agent response saved to file")
            if agent_response.success:
                # Combine structured data with agent insights
                enhanced_content = f"""RESUME ANALYSIS RESULTS

{analysis_summary}

STRATEGIC INSIGHTS:
{agent_response.content}

STRUCTURED DATA AVAILABLE:
- Personal Information: Available
- Technical Skills: {len(resume_analysis.technical_skills)} identified
- Work Experience: {len(resume_analysis.work_experience)} positions
- Education: {len(resume_analysis.education)} entries
- Certifications: {len(resume_analysis.certifications)} found
- Career Progression Analysis: {'Available' if resume_analysis.career_progression else 'Not Available'}

ANALYSIS CONFIDENCE: {resume_analysis.confidence_score:.1%} ({analysis_result.confidence_score:.1%})
PROCESSING TIME: {analysis_result.processing_time_seconds:.2f} seconds"""
                
                # with open("D:\\workspaces\\AI-Tutorials\\AI Agents\\MyAgentsTutorial\\agents\\2_openai\\Interview RolePlay\\interview_platform\\RAG\\Fifth_enhanced_content_AI_analysis_response.txt", "w", encoding="utf-8") as f:
                #     f.write(enhanced_content)
                # print(f"In analyze_resume method of Document Intelligence Agent: Enhanced content saved to file")
                # print(f"In analyze_resume method of Document Intelligence Agent: Stored in processing history. here is the enhanced content: {enhanced_content}")
                # Store in processing history
                self.processing_history.append({
                    "type": "resume_analysis",
                    "file_path": file_path,
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": resume_analysis.confidence_score,
                    "processing_time": analysis_result.processing_time_seconds
                })
                
                return AgentResponse(
                    success=True,
                    content=enhanced_content,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds + agent_response.processing_time,
                    metadata={
                        "resume_analysis": resume_analysis.dict(),
                        "analysis_metadata": analysis_result.metadata,
                        "structured_data_extracted": True,
                        "confidence_score": resume_analysis.confidence_score
                    }
                )
            else:
                # Return just the structured analysis if agent processing fails
                return AgentResponse(
                    success=True,
                    content=analysis_summary,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds,
                    metadata={
                        "resume_analysis": resume_analysis.dict(),
                        "agent_insights_failed": True,
                        "confidence_score": resume_analysis.confidence_score
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Resume analysis error: {e}")
            return AgentResponse(
                success=False,
                content=f"Resume analysis failed: {str(e)}",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                error=str(e)
            )
    
    async def analyze_job_description(self, 
                                    file_path: str,
                                    additional_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Analyze job description file and provide structured insights.
        
        Args:
            file_path: Path to job description file
            additional_context: Additional context for analysis
            
        Returns:
            AgentResponse with job description analysis results
        """
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent.analyze_job_description called: Analyze job description file and provide structured insights.")
        try:
            self.logger.info(f"Starting job description analysis: {Path(file_path).name}")
            
            if not self.document_service:
                return AgentResponse(
                    success=False,
                    content="Document Intelligence Service not available",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0,
                    error="Service initialization failed"
                )
            
            # Perform document analysis
            analysis_result = await self.document_service.analyze_job_description(file_path)
            
            if not analysis_result.success:
                return AgentResponse(
                    success=False,
                    content=f"Job description analysis failed: {analysis_result.error_message}",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds or 0.0,
                    error=analysis_result.error_message
                )
            
            # Convert result to JobDescriptionAnalysis object
            job_analysis = JobDescriptionAnalysis(**analysis_result.result_data)
            self.current_job_analysis = job_analysis
            
            # Generate human-readable analysis summary
            analysis_summary = self._generate_job_description_summary(job_analysis)
            
            # Create structured context
            context_data = {
                "document_type": "job_description",
                "file_path": file_path,
                "job_analysis": job_analysis.dict(),
                "analysis_metadata": analysis_result.metadata,
                "processing_time": analysis_result.processing_time_seconds
            }
            
            if additional_context:
                context_data.update(additional_context)
            
            # Process with agent for additional insights
            agent_response = await self.process_request(
                f"Please provide detailed insights and hiring recommendations based on this job description analysis: {analysis_summary[:500]}...",
                context_data
            )
            
            if agent_response.success:
                enhanced_content = f"""JOB DESCRIPTION ANALYSIS RESULTS

{analysis_summary}

HIRING STRATEGY INSIGHTS:
{agent_response.content}

STRUCTURED DATA EXTRACTED:
- Job Title: {job_analysis.job_title}
- Technical Requirements: {len(job_analysis.technical_requirements)}
- Experience Requirements: {len(job_analysis.experience_requirements)}
- Education Requirements: {len(job_analysis.education_requirements)}
- Key Responsibilities: {len(job_analysis.key_responsibilities)}
- Role Seniority: {job_analysis.role_seniority}
- Role Type: {job_analysis.role_type}

ANALYSIS CONFIDENCE: {job_analysis.confidence_score:.1%}
PROCESSING TIME: {analysis_result.processing_time_seconds:.2f} seconds"""
                
                # Store in processing history
                self.processing_history.append({
                    "type": "job_analysis", 
                    "file_path": file_path,
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": job_analysis.confidence_score,
                    "processing_time": analysis_result.processing_time_seconds
                })
                
                return AgentResponse(
                    success=True,
                    content=enhanced_content,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds + agent_response.processing_time,
                    metadata={
                        "job_analysis": job_analysis.dict(),
                        "analysis_metadata": analysis_result.metadata,
                        "structured_data_extracted": True,
                        "confidence_score": job_analysis.confidence_score
                    }
                )
            else:
                return AgentResponse(
                    success=True,
                    content=analysis_summary,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=analysis_result.processing_time_seconds,
                    metadata={
                        "job_analysis": job_analysis.dict(),
                        "agent_insights_failed": True,
                        "confidence_score": job_analysis.confidence_score
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Job description analysis error: {e}")
            return AgentResponse(
                success=False,
                content=f"Job description analysis failed: {str(e)}",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                error=str(e)
            )
    
    async def create_candidate_match_analysis(self) -> AgentResponse:
        """
        Create candidate-job matching analysis based on previously analyzed documents.
        
        Returns:
            AgentResponse with matching analysis results
        """
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent.create_candidate_match_analysis called: Create candidate-job matching analysis based on previously analyzed documents.")
        try:
            if not self.current_resume_analysis or not self.current_job_analysis:
                return AgentResponse(
                    success=False,
                    content="Both resume and job description must be analyzed first",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0,
                    error="Missing required document analyses"
                )
            
            self.logger.info("Creating candidate-job match analysis")
            
            # Generate candidate match using document service
            candidate_match = await self.document_service.create_candidate_match_analysis(
                self.current_resume_analysis,
                self.current_job_analysis
            )
            
            self.current_candidate_match = candidate_match
            
            # Generate human-readable match summary
            match_summary = self._generate_match_analysis_summary(candidate_match)
            
            # Create context for agent processing
            context_data = {
                "candidate_match": candidate_match.dict(),
                "resume_analysis": self.current_resume_analysis.dict(),
                "job_analysis": self.current_job_analysis.dict(),
                "analysis_type": "candidate_match"
            }
            
            # Get agent insights on the matching
            agent_response = await self.process_request(
                f"Please provide strategic hiring insights based on this candidate-job match analysis: {match_summary[:500]}...",
                context_data
            )
            
            if agent_response.success:
                enhanced_content = f"""CANDIDATE-JOB MATCH ANALYSIS

{match_summary}

STRATEGIC HIRING INSIGHTS:
{agent_response.content}

MATCH METRICS:
- Overall Match Score: {candidate_match.overall_match_score:.1%}
- Technical Skills Match: {len(candidate_match.technical_skills_match)} skills matched
- Experience Match Score: {candidate_match.experience_match_score:.1%}
- Education Match Score: {candidate_match.education_match_score:.1%}

KEY FINDINGS:
- Strong Matches: {len(candidate_match.strong_matches)} areas
- Skill Gaps: {len(candidate_match.skill_gaps)} identified
- Areas to Probe: {len(candidate_match.areas_to_probe)} recommended"""
                
                return AgentResponse(
                    success=True,
                    content=enhanced_content,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=agent_response.processing_time,
                    metadata={
                        "candidate_match": candidate_match.dict(),
                        "match_score": candidate_match.overall_match_score,
                        "analysis_complete": True
                    }
                )
            else:
                return AgentResponse(
                    success=True,
                    content=match_summary,
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0,
                    metadata={
                        "candidate_match": candidate_match.dict(),
                        "match_score": candidate_match.overall_match_score
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Candidate match analysis error: {e}")
            return AgentResponse(
                success=False,
                content=f"Candidate match analysis failed: {str(e)}",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                error=str(e)
            )
    
    def _generate_resume_summary(self, resume_analysis: ResumeAnalysis) -> str:
        """Generate human-readable resume analysis summary."""
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent._generate_resume_summary called: Generate human-readable resume analysis summary.")
        summary_parts = [
            f"CANDIDATE: {resume_analysis.personal_info.full_name or 'Name not extracted'}",
            ""
        ]
        
        # Technical skills summary
        if resume_analysis.technical_skills:
            skills_by_category = {}
            for skill in resume_analysis.technical_skills:
                category = skill.category
                if category not in skills_by_category:
                    skills_by_category[category] = []
                skills_by_category[category].append(skill.skill_name)
            
            summary_parts.append("TECHNICAL SKILLS:")
            for category, skills in skills_by_category.items():
                summary_parts.append(f"  {category}: {', '.join(skills)}")
            summary_parts.append("")
        
        # Work experience summary
        if resume_analysis.work_experience:
            summary_parts.append("WORK EXPERIENCE:")
            for exp in resume_analysis.work_experience[:3]:  # Top 3 experiences
                summary_parts.append(f"  {exp.job_title} at {exp.company_name} ({exp.duration})")
                if exp.key_responsibilities:
                    summary_parts.append(f"    Key: {exp.key_responsibilities[0][:100]}...")
            summary_parts.append("")
        
        # Career progression
        if resume_analysis.career_progression:
            cp = resume_analysis.career_progression
            summary_parts.extend([
                "CAREER PROGRESSION:",
                f"  Total Experience: {cp.total_experience_years} years",
                f"  Career Trajectory: {cp.career_trajectory}",
                f"  Industry Consistency: {'Yes' if cp.industry_consistency else 'No'}",
                ""
            ])
        
        # Strengths and gaps
        if resume_analysis.strengths:
            summary_parts.append("KEY STRENGTHS:")
            for strength in resume_analysis.strengths[:3]:
                summary_parts.append(f"  • {strength}")
            summary_parts.append("")
        
        if resume_analysis.potential_gaps:
            summary_parts.append("POTENTIAL GAPS:")
            for gap in resume_analysis.potential_gaps[:3]:
                summary_parts.append(f"  • {gap}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _generate_job_description_summary(self, job_analysis: JobDescriptionAnalysis) -> str:
        """Generate human-readable job description analysis summary."""
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent._generate_job_description_summary called: Generate human-readable job description analysis summary.")
        summary_parts = [
            f"POSITION: {job_analysis.job_title}",
            f"COMPANY: {job_analysis.company_info.company_name or 'Not specified'}",
            f"ROLE LEVEL: {job_analysis.role_seniority} {job_analysis.role_type}",
            ""
        ]
        
        # Technical requirements
        if job_analysis.technical_requirements:
            summary_parts.append("TECHNICAL REQUIREMENTS:")
            for req in job_analysis.technical_requirements[:5]:
                priority = req.priority.upper() if hasattr(req, 'priority') else 'REQUIRED'
                summary_parts.append(f"  {priority}: {req.requirement}")
            summary_parts.append("")
        
        # Experience requirements
        if job_analysis.experience_requirements:
            summary_parts.append("EXPERIENCE REQUIREMENTS:")
            for req in job_analysis.experience_requirements[:3]:
                summary_parts.append(f"  • {req.requirement}")
            summary_parts.append("")
        
        # Key responsibilities
        if job_analysis.key_responsibilities:
            summary_parts.append("KEY RESPONSIBILITIES:")
            for resp in job_analysis.key_responsibilities[:3]:
                summary_parts.append(f"  • {resp}")
            summary_parts.append("")
        
        # Success factors
        if job_analysis.critical_success_factors:
            summary_parts.append("CRITICAL SUCCESS FACTORS:")
            for factor in job_analysis.critical_success_factors[:3]:
                summary_parts.append(f"  • {factor}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _generate_match_analysis_summary(self, candidate_match: CandidateMatch) -> str:
        """Generate human-readable candidate match analysis summary."""
        print(f"document_intelligence_agent.py: Class DocumentIntelligenceAgent._generate_match_analysis_summary called: Generate human-readable candidate match analysis summary.")
        match_level = "Excellent" if candidate_match.overall_match_score >= 0.8 else \
                     "Good" if candidate_match.overall_match_score >= 0.6 else \
                     "Fair" if candidate_match.overall_match_score >= 0.4 else "Poor"
        
        summary_parts = [
            f"OVERALL MATCH: {match_level} ({candidate_match.overall_match_score:.1%})",
            ""
        ]
        
        # Strong matches
        if candidate_match.strong_matches:
            summary_parts.append("STRONG MATCHES:")
            for match in candidate_match.strong_matches[:5]:
                summary_parts.append(f"  ✓ {match}")
            summary_parts.append("")
        
        # Skill gaps
        if candidate_match.skill_gaps:
            summary_parts.append("SKILL GAPS:")
            for gap in candidate_match.skill_gaps[:5]:
                summary_parts.append(f"  ⚠ {gap}")
            summary_parts.append("")
        
        # Areas to probe
        if candidate_match.areas_to_probe:
            summary_parts.append("AREAS TO PROBE IN INTERVIEW:")
            for area in candidate_match.areas_to_probe[:5]:
                summary_parts.append(f"  🔍 {area}")
            summary_parts.append("")
        
        # Concerns
        if candidate_match.concerns:
            summary_parts.append("CONCERNS:")
            for concern in candidate_match.concerns[:3]:
                summary_parts.append(f"  ❌ {concern}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all current analysis results."""
        
        return {
            "resume_analysis": self.current_resume_analysis.dict() if self.current_resume_analysis else None,
            "job_analysis": self.current_job_analysis.dict() if self.current_job_analysis else None,
            "candidate_match": self.current_candidate_match.dict() if self.current_candidate_match else None,
            "processing_history": self.processing_history,
            "analysis_complete": all([
                self.current_resume_analysis,
                self.current_job_analysis,
                self.current_candidate_match
            ])
        }
    
    def clear_analysis_data(self):
        """Clear all current analysis data."""
        
        self.current_resume_analysis = None
        self.current_job_analysis = None
        self.current_candidate_match = None
        self.processing_history.clear()
        
        self.logger.info("Analysis data cleared")
    
    def convert_to_simplified_resume_analysis(self, resume_analysis: ResumeAnalysis) -> SimplifiedResumeAnalysis:
        """Convert a complex ResumeAnalysis to a SimplifiedResumeAnalysis."""
        try:
            # Convert personal info
            simplified_personal_info = SimplifiedPersonalInfo(
                full_name=resume_analysis.personal_info.full_name,
                email=resume_analysis.personal_info.email,
                location=resume_analysis.personal_info.location
            )
            
            # Convert technical skills (keep only top skills)
            simplified_technical_skills = []
            for skill in resume_analysis.technical_skills[:10]:  # Limit to top 10 skills
                simplified_skill = SimplifiedTechnicalSkill(
                    skill_name=skill.skill_name,
                    proficiency_level=skill.proficiency_level,
                    years_experience=skill.years_experience
                )
                simplified_technical_skills.append(simplified_skill)
            
            # Convert work experience (keep only key achievements)
            simplified_work_experience = []
            for exp in resume_analysis.work_experience:
                simplified_exp = SimplifiedWorkExperience(
                    job_title=exp.job_title,
                    company_name=exp.company_name,
                    duration=exp.duration,
                    key_achievements=exp.achievements[:3]  # Keep only top 3 achievements
                )
                simplified_work_experience.append(simplified_exp)
            
            # Convert education (keep only highest degree)
            highest_education = resume_analysis.education[0] if resume_analysis.education else None
            simplified_education = SimplifiedEducation(
                degree=f"{highest_education.degree} {highest_education.major}" if highest_education else "Not specified",
                institution=highest_education.institution if highest_education else "Not specified",
                graduation_year=highest_education.graduation_year if highest_education else None
            )
            
            # Get total experience years
            total_experience = resume_analysis.career_progression.total_experience_years if resume_analysis.career_progression else 0.0
            
            # Create simplified resume analysis
            simplified_resume = SimplifiedResumeAnalysis(
                personal_info=simplified_personal_info,
                technical_skills=simplified_technical_skills,
                work_experience=simplified_work_experience,
                education=simplified_education,
                total_experience_years=total_experience,
                key_strengths=resume_analysis.strengths[:5]  # Keep only top 5 strengths
            )
            
            return simplified_resume
            
        except Exception as e:
            self.logger.error(f"Failed to convert resume analysis to simplified format: {e}")
            raise
    
    async def analyze_resume_simplified(self, 
                                      file_path: str,
                                      additional_context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Analyze resume file and provide simplified structured insights.
        
        Args:
            file_path: Path to resume file
            additional_context: Additional context for analysis
            
        Returns:
            AgentResponse with simplified resume analysis results
        """
        try:
            self.logger.info(f"Starting simplified resume analysis: {Path(file_path).name}")
            
            # First get the full analysis
            full_analysis_result = await self.analyze_resume(file_path, additional_context)
            
            if not full_analysis_result.success:
                return full_analysis_result
            
            # Extract the full resume analysis
            if hasattr(full_analysis_result, 'metadata') and 'resume_analysis' in full_analysis_result.metadata:
                resume_data = full_analysis_result.metadata['resume_analysis']
            elif hasattr(full_analysis_result, 'data'):
                resume_data = full_analysis_result.data
            else:
                return AgentResponse(
                    success=False,
                    content="Resume analysis data not found",
                    agent_name=self.config.name,
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0,
                    error="Data extraction failed"
                )
            
            # Create ResumeAnalysis object
            clean_data = resume_data.copy()
            if 'analysis_timestamp' in clean_data:
                del clean_data['analysis_timestamp']
            
            full_resume_analysis = ResumeAnalysis(**clean_data)
            
            # Convert to simplified format
            simplified_resume = self.convert_to_simplified_resume_analysis(full_resume_analysis)
            
            # Generate simplified summary
            simplified_summary = self._generate_simplified_resume_summary(simplified_resume)
            
            # Create response
            return AgentResponse(
                success=True,
                content=simplified_summary,
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=full_analysis_result.processing_time,
                metadata={
                    "resume_analysis": simplified_resume.dict(),
                    "analysis_type": "simplified",
                    "original_analysis_available": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Simplified resume analysis failed: {e}")
            return AgentResponse(
                success=False,
                content=f"Simplified resume analysis failed: {str(e)}",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                error=str(e)
            )
    
    def _generate_simplified_resume_summary(self, resume: SimplifiedResumeAnalysis) -> str:
        """Generate a simplified human-readable resume summary."""
        summary_parts = []
        
        # Personal info
        summary_parts.append(f"CANDIDATE: {resume.personal_info.full_name}")
        summary_parts.append(f"LOCATION: {resume.personal_info.location}")
        summary_parts.append(f"EXPERIENCE: {resume.total_experience_years} years")
        summary_parts.append("")
        
        # Technical skills
        summary_parts.append("TECHNICAL SKILLS:")
        for skill in resume.technical_skills:
            level = skill.proficiency_level.value if skill.proficiency_level else "Unknown"
            years = f" ({skill.years_experience} years)" if skill.years_experience else ""
            summary_parts.append(f"- {skill.skill_name}: {level}{years}")
        summary_parts.append("")
        
        # Work experience
        summary_parts.append("WORK EXPERIENCE:")
        for exp in resume.work_experience:
            summary_parts.append(f"{exp.job_title} at {exp.company_name} ({exp.duration})")
            for achievement in exp.key_achievements:
                summary_parts.append(f"  • {achievement}")
        summary_parts.append("")
        
        # Education
        summary_parts.append("EDUCATION:")
        summary_parts.append(f"{resume.education.degree} from {resume.education.institution}")
        if resume.education.graduation_year:
            summary_parts.append(f"Graduated: {resume.education.graduation_year}")
        summary_parts.append("")
        
        # Key strengths
        if resume.key_strengths:
            summary_parts.append("KEY STRENGTHS:")
            for strength in resume.key_strengths:
                summary_parts.append(f"- {strength}")
        
        return "\n".join(summary_parts)


# Convenience functions
def create_document_intelligence_agent(name: str = "Document Intelligence Agent") -> DocumentIntelligenceAgent:
    """Create a document intelligence agent."""
    return DocumentIntelligenceAgent(name)


async def test_document_intelligence_agent():
    """Test the document intelligence agent functionality."""
    print("Testing Document Intelligence Agent...")
    print("=" * 60)
    
    try:
        # Create agent
        agent = DocumentIntelligenceAgent("Test Document Agent")
        print(f"✅ Created document intelligence agent")
        
        # Test basic functionality (without actual files)
        analysis_results = agent.get_analysis_results()
        print(f"✅ Analysis results retrieved")
        print(f"   Resume analysis: {'Available' if analysis_results['resume_analysis'] else 'None'}")
        print(f"   Job analysis: {'Available' if analysis_results['job_analysis'] else 'None'}")
        print(f"   Analysis complete: {analysis_results['analysis_complete']}")
        
        # Get service stats
        if agent.document_service:
            stats = agent.document_service.get_service_stats()
            print(f"\n📊 Document Service Statistics:")
            print(f"   Documents processed: {stats['documents_processed']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_document_intelligence_agent())
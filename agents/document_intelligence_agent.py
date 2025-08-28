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
        InterviewPlan, ProcessingResult, DocumentType
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
        try:
            self.document_service = DocumentIntelligenceService()
            self.logger.info("Document Intelligence Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Document Intelligence Service: {e}")
            self.document_service = None
        
        # Processing state
        self.current_resume_analysis = None
        self.current_job_analysis = None
        self.current_candidate_match = None
        self.processing_history = []
        
        self.logger.info(f"Document Intelligence Agent initialized: {name}")
    
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
            agent_response = await self.process_request(
                f"Please provide detailed insights and recommendations based on this resume analysis: {analysis_summary[:500]}...",
                context_data
            )
            
            if agent_response.success:
                # Combine structured data with agent insights
                enhanced_content = f"""RESUME ANALYSIS RESULTS

{analysis_summary}

STRATEGIC INSIGHTS:
{agent_response.content}

STRUCTURED DATA AVAILABLE:
- Personal Information: ✓
- Technical Skills: {len(resume_analysis.technical_skills)} identified
- Work Experience: {len(resume_analysis.work_experience)} positions
- Education: {len(resume_analysis.education)} entries
- Certifications: {len(resume_analysis.certifications)} found
- Career Progression Analysis: {'✓' if resume_analysis.career_progression else '❌'}

ANALYSIS CONFIDENCE: {resume_analysis.confidence_score:.1%} ({analysis_result.confidence_score:.1%})
PROCESSING TIME: {analysis_result.processing_time_seconds:.2f} seconds"""
                
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
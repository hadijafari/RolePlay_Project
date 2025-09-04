"""
Context Injection Service
Handles structured context creation and injection for different agent types.
This fixes the critical context passing issue in the current system.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from models.interview_models import (
        InterviewPlan, InterviewContext, InterviewState, DocumentAnalysisContext,
        InterviewPhase, ResponseEvaluation, Question, ProcessingResult
    )
except ImportError as e:
    print(f"Context Injection Service: Missing models: {e}")
    sys.exit(1)


class ContextType:
    """Context types for different agent operations."""
    STANDARD = "standard"
    INTERVIEW = "interview"
    DOCUMENT_ANALYSIS = "document_analysis"
    EVALUATION = "evaluation"


class ContextInjectionService:
    """
    Service for creating and managing structured contexts for agents.
    
    This service fixes the critical issue where interview plans and structured data
    were being flattened into unstructured text in the base agent's context creation.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Context templates and formatters
        self.context_formatters = {
            ContextType.STANDARD: self._format_standard_context,
            ContextType.INTERVIEW: self._format_interview_context,
            ContextType.DOCUMENT_ANALYSIS: self._format_document_context,
            ContextType.EVALUATION: self._format_evaluation_context
        }
        
        self.logger.info("Context Injection Service: Initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the context injection service."""
        logger = logging.getLogger("ContextInjectionService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Context Service: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_structured_context(self, 
                                user_input: str,
                                context_data: Dict[str, Any],
                                agent_type: str = "standard") -> Dict[str, Any]:
        """
        Create properly structured context based on agent type and data.
        
        This is the main method that fixes the context flattening problem.
        
        Args:
            user_input: User's input text
            context_data: Raw context data
            agent_type: Type of agent ("interview", "document", "evaluation", "standard")
            
        Returns:
            Structured context dictionary with proper formatting
        """
        try:
            # Determine context type based on available data
            context_type = self._determine_context_type(context_data, agent_type)
            
            # Get appropriate formatter
            formatter = self.context_formatters.get(context_type, self._format_standard_context)
            
            # Create structured context
            structured_context = formatter(user_input, context_data)
            
            # Add metadata
            structured_context["metadata"] = {
                "context_type": context_type,
                "creation_timestamp": datetime.now().isoformat(),
                "user_input_length": len(user_input),
                "context_data_keys": list(context_data.keys())
            }
            
            self.logger.debug(f"Created {context_type} context with {len(structured_context)} components")
            return structured_context
            
        except Exception as e:
            self.logger.error(f"Error creating structured context: {e}")
            # Fallback to basic context
            return self._create_fallback_context(user_input, context_data)
    
    def _determine_context_type(self, context_data: Dict[str, Any], agent_type: str) -> str:
        """Determine the appropriate context type based on available data."""
        
        # Check for interview-specific data
        if any(key in context_data for key in ["interview_plan", "interview_state", "current_section"]):
            return ContextType.INTERVIEW
        
        # Check for document analysis data
        if any(key in context_data for key in ["document_analysis", "resume_analysis", "job_analysis"]):
            return ContextType.DOCUMENT_ANALYSIS
        
        # Check for evaluation data
        if any(key in context_data for key in ["response_evaluation", "candidate_assessment"]):
            return ContextType.EVALUATION
        
        # Check agent type hint
        if agent_type in ["interview", "interviewer"]:
            return ContextType.INTERVIEW
        elif agent_type in ["document", "analysis"]:
            return ContextType.DOCUMENT_ANALYSIS
        elif agent_type in ["evaluation", "assessment"]:
            return ContextType.EVALUATION
        
        return ContextType.STANDARD
    
    def _format_interview_context(self, user_input: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format context for interview conductor agents.
        
        This creates proper structured context instead of flattening to text.
        """
        context = {
            "context_type": "interview",
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle interview plan
        if "interview_plan" in context_data:
            interview_plan = context_data["interview_plan"]
            
            if isinstance(interview_plan, InterviewPlan):
                context["interview_plan"] = {
                    "plan_id": interview_plan.plan_id,
                    "total_sections": len(interview_plan.interview_sections),
                    "estimated_duration": interview_plan.total_estimated_duration_minutes,
                    "key_focus_areas": interview_plan.key_focus_areas,
                    "evaluation_priorities": interview_plan.evaluation_priorities,
                    "sections": [
                        {
                            "phase": section.phase.value,
                            "name": section.section_name,
                            "objectives": section.objectives,
                            "duration": section.estimated_duration_minutes,
                            "questions_count": len(section.questions)
                        }
                        for section in interview_plan.interview_sections
                    ]
                }
            else:
                # Handle dict format
                context["interview_plan"] = interview_plan
        
        # Handle current interview state
        if "interview_state" in context_data:
            state = context_data["interview_state"]
            if isinstance(state, InterviewState):
                context["interview_state"] = {
                    "current_section": state.current_section_index,
                    "current_question": state.current_question_index,
                    "current_phase": state.current_phase.value,
                    "time_remaining": state.time_remaining_minutes,
                    "questions_asked_count": len(state.questions_asked),
                    "responses_count": len(state.responses_received)
                }
        
        # Handle current section info
        if "current_section" in context_data:
            context["current_section"] = context_data["current_section"]
        
        # Handle response history
        if "response_history" in context_data:
            context["response_history"] = context_data["response_history"]
        
        # Handle candidate match info
        if "candidate_match" in context_data:
            context["candidate_match"] = context_data["candidate_match"]
        
        # Create system instructions for interview agent
        context["system_instructions"] = self._generate_interview_instructions(context)
        
        return context
    
    def _format_document_context(self, user_input: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format context for document analysis agents."""
        context = {
            "context_type": "document_analysis",
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle document analysis results
        for key in ["resume_analysis", "job_analysis", "document_analysis"]:
            if key in context_data:
                context[key] = context_data[key]
        
        # Handle file information
        if "file_info" in context_data:
            context["file_info"] = context_data["file_info"]
        
        # Create system instructions for document analysis
        context["system_instructions"] = self._generate_document_analysis_instructions(context)
        
        return context
    
    def _format_evaluation_context(self, user_input: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format context for evaluation agents."""
        context = {
            "context_type": "evaluation",
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle evaluation data
        for key in ["response_evaluation", "candidate_assessment", "performance_metrics"]:
            if key in context_data:
                context[key] = context_data[key]
        
        # Create system instructions for evaluation
        context["system_instructions"] = self._generate_evaluation_instructions(context)
        
        return context
    
    def _format_standard_context(self, user_input: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format standard context for general agents."""
        context = {
            "context_type": "standard",
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add all context data as-is for standard agents
        context["additional_context"] = context_data
        
        return context
    
    def _generate_interview_instructions(self, context: Dict[str, Any]) -> str:
        """Generate specific system instructions for interview agents."""
        print("context_injection_service._generate_interview_instructions is called")
        instructions = [
            "You are an AI Interview Conductor. Your role is to conduct professional interviews based on the provided interview plan.",
            "",
            "INTERVIEW CONTEXT:"
        ]
        
        # Add interview plan context
        if "interview_plan" in context:
            plan = context["interview_plan"]
            instructions.extend([
                f"- Interview Duration: {plan.get('estimated_duration', 60)} minutes",
                f"- Total Sections: {plan.get('total_sections', 0)}",
                f"- Key Focus Areas: {', '.join(plan.get('key_focus_areas', []))}",
                f"- Evaluation Priorities: {', '.join(plan.get('evaluation_priorities', []))}",
                ""
            ])
        
        # Add current state context
        if "interview_state" in context:
            state = context["interview_state"]
            instructions.extend([
                "CURRENT INTERVIEW STATE:",
                f"- Current Phase: {state.get('current_phase', 'unknown')}",
                f"- Questions Asked: {state.get('questions_asked_count', 0)}",
                f"- Time Remaining: {state.get('time_remaining', 'unknown')} minutes",
                ""
            ])
        
        instructions.extend([
            "INSTRUCTIONS:",
            "1. Ask the EXACT question provided in the NEXT_QUESTION field",
            "2. You may add a brief transition, but ask the PROVIDED question",
            "3. Listen carefully to responses and ask relevant follow-up questions",
            "4. After follow-ups (max 2) or good answers, move to next question",
            "5. When NEXT_QUESTION says [NO MORE QUESTIONS], wrap up the interview",
            "6. Maintain a friendly but professional tone throughout",
            "",
            "NEXT QUESTION TO ASK:"
        ])
        
        # Add the specific next question
        if "NEXT_QUESTION" in context:
            instructions.append(f"[NEXT_QUESTION]: {context['NEXT_QUESTION']}")
            
            if "current_question_info" in context and context["current_question_info"]:
                info = context["current_question_info"]
                instructions.extend([
                    f"Question {info.get('question_number', 0)} of {info.get('total_in_section', 0)} in current section",
                    f"Type: {info.get('question_type', 'unknown')}",
                    f"Focus: {', '.join(info.get('skill_focus', []))}",
                    ""
                ])

        print(f"Here is the system prompt interview instructions: {instructions}")
        return "\n".join(instructions)
    
    def _generate_document_analysis_instructions(self, context: Dict[str, Any]) -> str:
        """Generate system instructions for document analysis agents."""
        print("context_injection_service._generate_document_analysis_instructions is called")
        instructions = [
            "You are a Document Analysis AI. Your role is to extract and analyze information from resumes and job descriptions.",
            "",
            "ANALYSIS OBJECTIVES:",
            "1. Extract structured data from documents",
            "2. Identify key skills, experience, and qualifications",
            "3. Assess candidate-job fit",
            "4. Provide insights for interview planning",
            "",
            "ANALYSIS CONTEXT:"
        ]
        
        # Add document-specific context
        if "file_info" in context:
            file_info = context["file_info"]
            instructions.append(f"- Document Type: {file_info.get('type', 'unknown')}")
            instructions.append(f"- File Size: {file_info.get('size', 'unknown')}")
        
        instructions.extend([
            "",
            "Provide detailed analysis in structured format."
        ])
        
        return "\n".join(instructions)
    
    def _generate_evaluation_instructions(self, context: Dict[str, Any]) -> str:
        """Generate system instructions for evaluation agents."""
        print("context_injection_service._generate_evaluation_instructions is called")
        return """You are an Interview Evaluation AI. Your role is to assess candidate responses and provide feedback.

EVALUATION CRITERIA:
1. Technical accuracy and depth
2. Communication clarity
3. Problem-solving approach
4. Cultural fit indicators
5. Overall competency for the role

Provide constructive feedback and scoring based on the response context."""
    
    def _create_fallback_context(self, user_input: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback context if structured context creation fails."""
        self.logger.warning("Using fallback context creation")
        
        return {
            "context_type": "fallback",
            "user_input": user_input,
            "raw_context_data": context_data,
            "timestamp": datetime.now().isoformat(),
            "system_instructions": "You are a helpful AI assistant. Respond based on the provided context."
        }
    


    def convert_to_agent_messages(self, structured_context: Dict[str, Any], 
                                conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Convert structured context to OpenAI message format for agents.
        
        This method ensures proper context injection into the agent's conversation.
        """
        print("context_injection_service.convert_to_agent_messages is called")
        messages = []
        
        # Add system message with instructions
        if "system_instructions" in structured_context:
            messages.append({
                "role": "system",
                "content": structured_context["system_instructions"]
            })
        
        # Add context as system message if it's interview context
        if structured_context.get("context_type") == "interview":
            context_summary = self._create_interview_context_summary(structured_context)
            messages.append({
                "role": "system", 
                "content": f"INTERVIEW CONTEXT:\n{context_summary}"
            })
        
        # Add conversation history (all previous Q&As)
        if conversation_history:
            for msg in conversation_history:
                # Skip system messages from history to avoid duplicates
                if msg["role"] != "system":
                    messages.append(msg)
        
        # Add current user input with follow-up context if available
        current_message = structured_context["user_input"]
        
        # Add follow-up tracking info to user message
        if structured_context.get("context_type") == "interview" and "followup_context" in structured_context:
            followup_info = structured_context.get("followup_context", {})
            followup_count = followup_info.get("current_followup_count", 0)
            max_followups = followup_info.get("max_followups_allowed", 2)
            
            if followup_count > 0:
                current_message = f"[Follow-up {followup_count}/{max_followups}]\n{current_message}"
        
        messages.append({
            "role": "user",
            "content": current_message
        })

        print(f"Here is the messages for the user prompt for this section: {messages}")
        return messages




    # def convert_to_agent_messages(self, structured_context: Dict[str, Any]) -> List[Dict[str, str]]:
    #     """
    #     Convert structured context to OpenAI message format for agents.
        
    #     This method ensures proper context injection into the agent's conversation.
    #     """
    #     print("context_injection_service.convert_to_agent_messages is called")
    #     messages = []
        
    #     # Add system message with instructions
    #     if "system_instructions" in structured_context:
    #         messages.append({
    #             "role": "system",
    #             "content": structured_context["system_instructions"]
    #         })
        
    #     # Add context as system message if it's interview context
    #     if structured_context.get("context_type") == "interview":
    #         context_summary = self._create_interview_context_summary(structured_context)
    #         messages.append({
    #             "role": "system", 
    #             "content": f"INTERVIEW CONTEXT:\n{context_summary}"
    #         })
        
    #     # Add user input
    #     messages.append({
    #         "role": "user",
    #         "content": structured_context["user_input"]
    #     })
        

    #     print(f"Here is the messages for the user prompt for this section: {messages}")
    #     return messages
    
    def _create_interview_context_summary(self, context: Dict[str, Any]) -> str:
        """Create a concise summary of interview context for the agent."""
        print("context_injection_service._create_interview_context_summary is called")
        summary_parts = []
        
        # Interview plan summary
        if "interview_plan" in context:
            plan = context["interview_plan"]
            summary_parts.append(f"Interview Plan: {plan.get('total_sections', 0)} sections, focusing on {', '.join(plan.get('key_focus_areas', []))}")
        
        # Current state summary
        if "interview_state" in context:
            state = context["interview_state"]
            summary_parts.append(f"Current: {state.get('current_phase', 'unknown')} phase, {state.get('questions_asked_count', 0)} questions asked")
        
        # Candidate match info
        if "candidate_match" in context:
            summary_parts.append("Candidate profile and job matching data available")
        
        return "\n".join(summary_parts) if summary_parts else "Standard interview context"
    
    def validate_context_structure(self, context: Dict[str, Any]) -> ProcessingResult:
        """Validate that the context structure is properly formatted."""
        print("context_injection_service.validate_context_structure is called")
        try:
            required_fields = ["context_type", "user_input", "timestamp"]
            missing_fields = [field for field in required_fields if field not in context]
            
            if missing_fields:
                return ProcessingResult(
                    success=False,
                    error_message=f"Missing required context fields: {missing_fields}"
                )
            
            # Type-specific validation
            context_type = context.get("context_type")
            
            if context_type == "interview":
                if "interview_plan" not in context and "interview_state" not in context:
                    return ProcessingResult(
                        success=False,
                        error_message="Interview context missing interview_plan or interview_state"
                    )
            
            return ProcessingResult(
                success=True,
                result_data={"validation": "passed", "context_type": context_type}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Context validation error: {e}"
            )
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context creation operations."""
        print("context_injection_service.get_context_statistics is called")
        # This would be enhanced with actual statistics tracking
        return {
            "service_status": "active",
            "supported_context_types": list(self.context_formatters.keys()),
            "last_update": datetime.now().isoformat()
        }


# Convenience functions for common context operations
def create_interview_context(interview_plan: InterviewPlan, 
                           interview_state: Optional[InterviewState] = None,
                           user_input: str = "") -> Dict[str, Any]:
    """Convenience function to create interview context."""
    print("create_interview_context is called")
    service = ContextInjectionService()
    context_data = {
        "interview_plan": interview_plan
    }
    
    if interview_state:
        context_data["interview_state"] = interview_state
    
    return service.create_structured_context(user_input, context_data, "interview")


def create_document_analysis_context(document_analysis: Dict[str, Any],
                                   user_input: str = "") -> Dict[str, Any]:
    """Convenience function to create document analysis context."""
    print("create_document_analysis_context is called")
    service = ContextInjectionService()
    return service.create_structured_context(user_input, {"document_analysis": document_analysis}, "document")


# Export main classes and functions
__all__ = [
    'ContextInjectionService',
    'ContextType', 
    'create_interview_context',
    'create_document_analysis_context'
]
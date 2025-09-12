"""
Enhanced Feedback Agent for Interview Analysis
Provides technical feedback on interview questions and answers
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
import os
from pathlib import Path

# Setup logging
logger = logging.getLogger("FEEDBACK_AGENT")

class FeedbackAgent:
    """
    Async feedback agent that analyzes interview Q&A pairs and provides
    technical feedback including strengths, weaknesses, and ideal answers.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the feedback agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.openai_client = None
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it to constructor.")
        
        try:
            self.openai_client = OpenAI(api_key=self.api_key)
            logger.info("Feedback Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def generate_feedback(self, question: str, answer: str, question_number: int = 1) -> Dict[str, Any]:
        """
        Generate comprehensive feedback for a Q&A pair.
        
        Args:
            question: The interviewer's question
            answer: The interviewee's answer
            question_number: The question number in the interview
            
        Returns:
            Dict containing feedback analysis
        """
        try:
            logger.info(f"Generating feedback for question {question_number}")
            
            # Create the system prompt for technical feedback
            system_prompt = self._get_system_prompt()
            
            # Create the user message with context
            user_message = self._create_user_message(question, answer, question_number)
            
            # Make async API call
            response = await self._call_openai_async(system_prompt, user_message)
            
            # Parse the response
            feedback_data = self._parse_feedback_response(response)
            
            # Add metadata
            feedback_data.update({
                "question_number": question_number,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer
            })
            
            logger.info(f"Feedback generated successfully for question {question_number}")
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error generating feedback for question {question_number}: {e}")
            return self._get_fallback_feedback(question, answer, question_number)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the feedback agent."""
        return """You are an expert technical interviewer and career coach with deep knowledge across multiple technical domains including software engineering, data science, AI/ML, cybersecurity, cloud computing, and system design.

Your role is to provide comprehensive, constructive feedback on interview Q&A pairs. For each question and answer, you must analyze:

1. **Technical Accuracy**: Is the answer technically correct?
2. **Completeness**: Does the answer address all parts of the question?
3. **Clarity**: Is the answer clear and well-structured?
4. **Depth**: Does the answer show appropriate depth of knowledge?
5. **Practical Experience**: Does the answer demonstrate real-world experience?
6. **Communication Skills**: How well is the answer communicated?

For each analysis, provide:
- **Strengths**: What the candidate did well
- **Weaknesses**: Areas that need improvement
- **Ideal Answer**: What a strong answer would look like
- **Technical Assessment**: Professional evaluation of technical knowledge
- **Improvement Suggestions**: Specific actionable advice

Be constructive, professional, and specific. Focus on helping the candidate improve while being honest about gaps in knowledge or communication.

Format your response as JSON with these exact keys:
{
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "ideal_answer": "Detailed ideal answer that addresses the question comprehensively",
    "technical_assessment": "Professional technical evaluation",
    "improvement_suggestions": ["suggestion1", "suggestion2", ...],
    "overall_score": 0.85,
    "summary": "Brief overall assessment"
}"""
    
    def _create_user_message(self, question: str, answer: str, question_number: int) -> str:
        """Create the user message for the API call."""
        return f"""Please analyze this interview Q&A pair and provide comprehensive feedback:

**Question {question_number}:**
{question}

**Answer:**
{answer}

Please provide detailed technical feedback including strengths, weaknesses, ideal answer, and improvement suggestions. Be specific and constructive in your analysis."""
    
    async def _call_openai_async(self, system_prompt: str, user_message: str) -> str:
        """Make async call to OpenAI API."""
        try:
            # Run the OpenAI call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_feedback_response(self, response: str) -> Dict[str, Any]:
        """Parse the OpenAI response into structured feedback."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                feedback = json.loads(json_str)
                
                # Validate required keys
                required_keys = ["strengths", "weaknesses", "ideal_answer", "technical_assessment", "improvement_suggestions"]
                for key in required_keys:
                    if key not in feedback:
                        feedback[key] = "Not provided"
                
                # Ensure overall_score is a float
                if "overall_score" not in feedback:
                    feedback["overall_score"] = 0.5
                else:
                    try:
                        feedback["overall_score"] = float(feedback["overall_score"])
                    except (ValueError, TypeError):
                        feedback["overall_score"] = 0.5
                
                return feedback
            else:
                logger.warning("No JSON found in response, using fallback")
                return self._get_fallback_feedback("", "", 1)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._get_fallback_feedback("", "", 1)
    
    def _get_fallback_feedback(self, question: str, answer: str, question_number: int) -> Dict[str, Any]:
        """Provide fallback feedback when AI fails."""
        return {
            "strengths": ["Attempted to answer the question", "Showed engagement"],
            "weaknesses": ["Answer could be more detailed", "Consider providing specific examples"],
            "ideal_answer": "A comprehensive answer that directly addresses the question with specific examples and technical details.",
            "technical_assessment": "Unable to assess due to technical issues. Please try again.",
            "improvement_suggestions": [
                "Provide more specific examples",
                "Structure your answer clearly",
                "Include technical details when relevant"
            ],
            "overall_score": 0.5,
            "summary": "Feedback generation encountered technical issues. Please try again.",
            "question_number": question_number,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "is_fallback": True
        }
    
    def format_feedback_for_display(self, feedback: Dict[str, Any]) -> str:
        """Format feedback data for display in the conversation log."""
        try:
            formatted = f"""
**FEEDBACK FOR QUESTION {feedback.get('question_number', 'N/A')}:**

**Strengths:**
{chr(10).join(f"• {strength}" for strength in feedback.get('strengths', []))}

**Weaknesses:**
{chr(10).join(f"• {weakness}" for weakness in feedback.get('weaknesses', []))}

**Ideal Answer:**
{feedback.get('ideal_answer', 'Not provided')}

**Technical Assessment:**
{feedback.get('technical_assessment', 'Not provided')}

**Improvement Suggestions:**
{chr(10).join(f"• {suggestion}" for suggestion in feedback.get('improvement_suggestions', []))}

**Overall Score:** {feedback.get('overall_score', 0.5):.2f}/1.0
**Summary:** {feedback.get('summary', 'No summary available')}
"""
            return formatted.strip()
            
        except Exception as e:
            logger.error(f"Error formatting feedback: {e}")
            return f"Error formatting feedback: {str(e)}"
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about feedback generation."""
        return {
            "agent_initialized": self.openai_client is not None,
            "api_key_configured": bool(self.api_key),
            "timestamp": datetime.now().isoformat()
        }


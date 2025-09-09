"""
Feedback Agent
Provides concise, evidence-based feedback on a candidate's answer: strengths and weaknesses.
Runs on the side; do not block the interview flow.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from .base_agent import BaseAgent, AgentConfig, AgentResponse


class FeedbackAgent(BaseAgent):
    """Agent that critiques candidate answers with strengths and weaknesses."""

    def __init__(self, name: str = "Feedback Agent", model: str = "gpt-4o-mini"):
        instructions = (
            "You are an interview feedback assistant. "
            "You will receive the interview question and the candidate's answer. "
            "Provide objective, evidence-based feedback with two sections: \n"
            "- Strengths: bullet points of what the answer does well, grounded in the answer's content.\n"
            "- Weaknesses: bullet points of any gaps, inaccuracies, missing details, or vague statements.\n"
            "Do not invent facts. If information is missing, state it as missing rather than assuming.\n"
            "Keep output concise (3-6 bullets total). Focus on technical rigor, clarity, and relevance."
        )

        config = AgentConfig(
            name=name,
            instructions=instructions,
            model=model,
            timeout=30.0,
            agent_type="evaluation",
            log_level="INFO",
        )

        super().__init__(config)
        self.logger = logging.getLogger("FeedbackAgent")

    async def evaluate(self, question_text: str, answer_text: str) -> AgentResponse:
        """Evaluate a candidate's answer against a question."""
        user_prompt = (
            "Question:\n" + question_text.strip() + "\n\n" +
            "Candidate Answer:\n" + answer_text.strip() + "\n\n" +
            "Provide feedback now."
        )

        result = await self.process_request(user_prompt)
        return result


__all__ = ["FeedbackAgent"]



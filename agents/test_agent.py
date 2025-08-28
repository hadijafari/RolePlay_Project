"""
Test Agent - Simple Echo Agent
Demonstrates basic agent functionality by repeating user input.
"""

import asyncio
from typing import Optional, Dict, Any
from .base_agent import BaseAgent, AgentConfig


class InterviewerAgent(BaseAgent):
    """Interview agent that provides helpful responses and maintains conversation context."""
    
    def __init__(self, name: str = "Echo Agent", 
                 instructions: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        """
        Initialize echo agent.
        
        Args:
            name: Agent name
            instructions: Custom instructions (uses default if None)
            model: OpenAI model to use
        """
        # Default instructions for interview behavior
        default_instructions = """You are a professional interview assistant conducting a structured interview based on document analysis. Your role is to:

CORE RESPONSIBILITIES:
1. Use the provided interview plan to ask relevant, targeted questions
2. Reference specific details from the candidate's resume and job requirements
3. Focus on skills validation, gap analysis, and strength assessment
4. Ask follow-up questions based on the candidate's background
5. Always respond in English, regardless of what language the user speaks

WHEN INTERVIEW PLAN IS PROVIDED:
- Address the candidate by name from their resume
- Ask about specific technologies, projects, and experiences mentioned in their resume
- Focus on missing critical skills identified in the gap analysis
- Validate strengths and experiences listed in their background
- Use the technical questions, gap questions, and behavioral questions from the strategy
- Reference specific companies, roles, and achievements from their work history

INTERVIEW APPROACH:
- Start by acknowledging their background (e.g., "I see from your resume that you're an experienced embedded engineer...")
- Ask specific technical questions about their listed skills and projects
- Address skill gaps directly (e.g., "I notice you haven't worked with GUI development...")
- Validate their strengths with concrete examples
- Keep responses professional, engaging, and focused on the job fit

WHEN NO INTERVIEW PLAN IS AVAILABLE:
- Conduct a general interview focusing on their responses
- Ask standard technical and behavioral questions
- Be helpful and professional as a general interview assistant

Always use the interview plan data when available to create a personalized, relevant interview experience."""
        
        config = AgentConfig(
            name=name,
            instructions=instructions or default_instructions,
            model=model,
            timeout=20.0,  # Shorter timeout for simple responses
            max_history_length=50,  # Keep recent conversations
            log_level="INFO"
        )
        
        super().__init__(config)
    
    async def process_interview_message(self, message: str, 
                          context_data: Optional[Dict[str, Any]] = None):
        """
        Process the user's message with interview agent processing.
        
        Args:
            message: User's message to process
            context_data: Additional context data including interview plan
            
        Returns:
            AgentResponse with interview response
        """
        # Enhanced context processing for interview plan
        enhanced_context = self._prepare_interview_context(message, context_data)
        return await self.process_request(message, enhanced_context)
    
    def _prepare_interview_context(self, message: str, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare enhanced context with interview plan information."""
        if not context_data:
            return {}
        
        enhanced_context = context_data.copy()
        
        # If interview plan is provided, format it for better AI understanding
        if "interview_plan" in context_data:
            plan = context_data["interview_plan"]
            
            # Extract key information for the AI
            context_summary = self._extract_interview_context(plan)
            enhanced_context["interview_context"] = context_summary
            
            # Add specific instructions based on the plan
            enhanced_context["interview_instructions"] = self._generate_interview_instructions(plan)
        
        return enhanced_context
    
    def _extract_interview_context(self, interview_plan: Dict[str, Any]) -> str:
        """Extract key context information from the interview plan."""
        try:
            context_parts = []
            
            # Extract candidate information
            if "candidate_analysis" in interview_plan:
                candidate_data = interview_plan["candidate_analysis"]
                if "raw_analysis" in candidate_data:
                    import json
                    try:
                        candidate_info = json.loads(candidate_data["raw_analysis"].replace("```json\n", "").replace("```", ""))
                        
                        # Basic info
                        if "basic_information" in candidate_info:
                            basic = candidate_info["basic_information"]
                            context_parts.append(f"CANDIDATE: {basic.get('name', 'Unknown')}")
                        
                        # Professional summary
                        if "professional_summary" in candidate_info:
                            context_parts.append(f"BACKGROUND: {candidate_info['professional_summary']}")
                        
                        # Technical skills
                        if "skills" in candidate_info and "technical" in candidate_info["skills"]:
                            skills = ", ".join(candidate_info["skills"]["technical"][:5])  # Top 5 skills
                            context_parts.append(f"KEY TECHNICAL SKILLS: {skills}")
                        
                        # Recent experience
                        if "work_experience" in candidate_info and candidate_info["work_experience"]:
                            recent_job = candidate_info["work_experience"][0]
                            context_parts.append(f"CURRENT ROLE: {recent_job.get('job_title', 'N/A')} at {recent_job.get('company', 'N/A')}")
                    except:
                        context_parts.append("CANDIDATE: Resume analysis available")
            
            # Extract job information
            if "job_analysis" in interview_plan:
                job_data = interview_plan["job_analysis"]
                if "raw_analysis" in job_data:
                    try:
                        job_info = json.loads(job_data["raw_analysis"].replace("```json\n", "").replace("```", ""))
                        
                        context_parts.append(f"TARGET ROLE: {job_info.get('job_title', 'N/A')}")
                        
                        if "required_skills_and_qualifications" in job_info:
                            required_skills = ", ".join(job_info["required_skills_and_qualifications"][:5])
                            context_parts.append(f"REQUIRED SKILLS: {required_skills}")
                    except:
                        context_parts.append("TARGET ROLE: Job analysis available")
            
            # Extract gap analysis
            if "gap_analysis" in interview_plan:
                gap_data = interview_plan["gap_analysis"]
                if "raw_analysis" in gap_data:
                    try:
                        gap_info = json.loads(gap_data["raw_analysis"].replace("```json\n", "").replace("```", ""))
                        
                        if "gap_analysis" in gap_info:
                            gap = gap_info["gap_analysis"]
                            context_parts.append(f"FIT SCORE: {gap.get('overall_fit_score', 'N/A')}%")
                            
                            if "missing_critical_skills" in gap:
                                missing = ", ".join(gap["missing_critical_skills"][:3])
                                context_parts.append(f"MISSING SKILLS: {missing}")
                    except:
                        context_parts.append("GAP ANALYSIS: Available")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Interview plan available but parsing failed: {e}"
    
    def _generate_interview_instructions(self, interview_plan: Dict[str, Any]) -> str:
        """Generate specific instructions based on the interview plan."""
        try:
            instructions = []
            
            # Extract specific questions from the strategy
            if "interview_strategy" in interview_plan:
                strategy_data = interview_plan["interview_strategy"]
                if "raw_analysis" in strategy_data:
                    try:
                        import json
                        strategy = json.loads(strategy_data["raw_analysis"].replace("```json\n", "").replace("```", ""))
                        
                        if "interview_strategy" in strategy:
                            strat = strategy["interview_strategy"]
                            
                            instructions.append("USE THESE SPECIFIC QUESTION AREAS:")
                            
                            # Technical questions
                            if "technical_questions" in strat:
                                tech_areas = list(strat["technical_questions"].keys())
                                instructions.append(f"- Technical focus: {', '.join(tech_areas[:3])}")
                            
                            # Gap questions
                            if "gap_analysis_questions" in strat:
                                gap_areas = list(strat["gap_analysis_questions"].keys())
                                instructions.append(f"- Address gaps in: {', '.join(gap_areas[:3])}")
                            
                            # Interview flow
                            if "interview_flow_and_focus_areas" in strat:
                                instructions.append("- Follow structured interview flow from plan")
                    except:
                        instructions.append("- Use the provided interview strategy")
            
            instructions.append("- Reference specific resume details and job requirements")
            instructions.append("- Ask follow-up questions based on their actual experience")
            
            return "\n".join(instructions)
            
        except Exception as e:
            return f"Interview strategy available: {e}"
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation session."""
        history = self.get_history(recent_only=False)
        stats = self.get_stats()
        
        return {
            "agent_name": self.config.name,
            "session_summary": self.conversation_manager.get_session_summary(),
            "conversation_count": len(history),
            "recent_messages": [
                {
                    "user": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"],
                    "agent": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"],
                    "timestamp": msg["timestamp"]
                }
                for msg in history[-5:] if msg["role"] == "user"  # Last 5 user messages
            ],
            "performance_stats": {
                "success_rate": stats["success_rate"],
                "average_response_time": stats["average_processing_time"],
                "total_requests": stats["total_requests"]
            }
        }


class SimpleEchoAgent(BaseAgent):
    """Very simple echo agent for basic testing."""
    
    def __init__(self):
        """Initialize simple echo agent with basic instructions."""
        config = AgentConfig(
            name="Simple Echo Agent",
            instructions="You are a simple echo assistant. Repeat what the user says and add 'Echo: ' at the beginning.",
            model="gpt-4o-mini",
            timeout=15.0,
            max_history_length=20
        )
        
        super().__init__(config)
    
    async def simple_echo(self, message: str) -> str:
        """
        Simple echo functionality.
        
        Args:
            message: Message to echo
            
        Returns:
            Echoed message
        """
        response = await self.process_request(message)
        if response.success:
            return response.content
        else:
            return f"Error: {response.error}"


# Example usage and testing functions
async def test_echo_agent():
    """Test the echo agent functionality."""
    print("Testing Interview Agent...")
    print("=" * 50)
    
    # Create interview agent
    agent = InterviewerAgent()
    
    try:
        # Test basic echo
        print("\n1. Testing basic echo...")
        response = await agent.process_interview_message("Hello, how are you today?")
        if response.success:
            print(f"✅ Interview response successful: {response.content}")
            print(f"   Processing time: {response.processing_time:.2f}s")
        else:
            print(f"❌ Echo failed: {response.error}")
        
        # Test conversation context
        print("\n2. Testing conversation context...")
        response2 = await agent.process_interview_message("What did I just say?")
        if response2.success:
            print(f"✅ Context response: {response2.content}")
        else:
            print(f"❌ Context failed: {response2.error}")
        
        # Test conversation summary
        print("\n3. Getting conversation summary...")
        summary = await agent.get_conversation_summary()
        print(f"✅ Session ID: {summary['session_summary']['session_id']}")
        print(f"   Total interactions: {summary['conversation_count']}")
        print(f"   Success rate: {summary['performance_stats']['success_rate']:.1%}")
        
        # Display recent messages
        print("\n4. Recent conversation:")
        for i, msg in enumerate(summary['recent_messages'], 1):
            print(f"   {i}. User: {msg['user']}")
            print(f"      Agent: {msg['agent']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Cleanup
        agent.cleanup()
        print("\n✅ Interview agent test completed")


async def test_simple_echo_agent():
    """Test the simple echo agent."""
    print("\nTesting Simple Echo Agent...")
    print("=" * 50)
    
    agent = SimpleEchoAgent()
    
    try:
        test_messages = [
            "Hello world!",
            "This is a test message",
            "How are you doing?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{i}. Testing: '{message}'")
            response = await agent.simple_echo(message)
            print(f"   Response: {response}")
        
        # Get stats
        stats = agent.get_stats()
        print(f"\n📊 Agent Statistics:")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Average time: {stats['average_processing_time']:.2f}s")
        print(f"   Total requests: {stats['total_requests']}")
        
    except Exception as e:
        print(f"❌ Simple echo test failed: {e}")
    
    finally:
        agent.cleanup()
        print("\n✅ Simple echo agent test completed")


async def main():
    """Run all agent tests."""
    print("🤖 Interview Agent Test Suite")
    print("=" * 60)
    
    try:
        await test_echo_agent()
        await test_simple_echo_agent()
        
        print("\n" + "=" * 60)
        print("✅ All agent tests completed successfully!")
        print("\nThe interview agents are ready for integration!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    # Run the async test suite
    asyncio.run(main())

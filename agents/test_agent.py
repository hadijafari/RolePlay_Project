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
        default_instructions = """You are a helpful interview assistant. Your role is to:
1. Listen to what the user says
2. Provide thoughtful and relevant responses
3. Ask follow-up questions when appropriate
4. Keep responses concise and engaging
5. Maintain conversation context from previous interactions
6. Always respond in English, regardless of what language the user speaks

Always be polite, professional, and helpful in your responses."""
        
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
            context_data: Additional context data
            
        Returns:
            AgentResponse with interview response
        """
        return await self.process_request(message, context_data)
    
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

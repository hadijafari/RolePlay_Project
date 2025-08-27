"""
Agent Service - Centralized Agent Management
Manages multiple AI agents and provides a unified interface for agent interactions.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

try:
    # Add the project root to Python path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    sys.path.insert(0, project_root)
    
    # Debug: Print the paths
    print(f"Current file directory: {current_file_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path after modification: {sys.path[0]}")
    
    from agents.base_agent import BaseAgent, AgentConfig, AgentResponse
    from agents.test_agent import InterviewerAgent, SimpleEchoAgent
    print("✅ Agent imports successful")
except ImportError as e:
    print(f"Agent Service: Missing required dependency: {e}")
    print("Please ensure all agent modules are available")
    sys.exit(1)


class AgentService:
    """Centralized service for managing multiple AI agents."""
    
    def __init__(self):
        """Initialize the agent service."""
        self.logger = self._setup_logger()
        self.agents: Dict[str, BaseAgent] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.service_stats = {
            "total_agent_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "agents_created": 0,
            "sessions_started": 0
        }
        
        self.logger.info("Agent Service: Initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent service."""
        logger = logging.getLogger("AgentService")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Agent Service: %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_agent(self, agent_type: str, name: str, 
                    instructions: Optional[str] = None,
                    model: str = "gpt-4o-mini") -> Optional[BaseAgent]:
        """
        Create a new agent of the specified type.
        
        Args:
            agent_type: Type of agent to create ('echo', 'simple_echo', 'custom')
            name: Name for the agent
            instructions: Custom instructions (optional)
            model: OpenAI model to use
            
        Returns:
            Created agent instance or None if creation failed
        """
        try:
            if agent_type.lower() == "echo":
                agent = InterviewerAgent(name=name, instructions=instructions, model=model)
            elif agent_type.lower() == "simple_echo":
                agent = SimpleEchoAgent()
            elif agent_type.lower() == "custom":
                config = AgentConfig(
                    name=name,
                    instructions=instructions or "You are a helpful assistant",
                    model=model
                )
                agent = BaseAgent(config)
            else:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return None
            
            # Store the agent
            self.agents[name] = agent
            self.service_stats["agents_created"] += 1
            
            self.logger.info(f"Created {agent_type} agent: {name}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent '{name}': {e}")
            return None
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents with their information."""
        agent_list = []
        
        for name, agent in self.agents.items():
            stats = agent.get_stats()
            agent_info = {
                "name": name,
                "type": agent.__class__.__name__,
                "model": agent.config.model,
                "instructions": agent.config.instructions[:100] + "..." if len(agent.config.instructions) > 100 else agent.config.instructions,
                "status": "active",
                "stats": {
                    "total_requests": stats["total_requests"],
                    "success_rate": f"{stats['success_rate']:.1%}",
                    "average_time": f"{stats['average_processing_time']:.2f}s"
                }
            }
            agent_list.append(agent_info)
        
        return agent_list
    
    async def process_request(self, agent_name: str, user_input: str,
                           context_data: Optional[Dict[str, Any]] = None) -> Optional[AgentResponse]:
        """
        Process a request using the specified agent.
        
        Args:
            agent_name: Name of the agent to use
            user_input: User's input text
            context_data: Additional context data
            
        Returns:
            Agent response or None if processing failed
        """
        try:
            agent = self.get_agent(agent_name)
            if not agent:
                self.logger.error(f"Agent not found: {agent_name}")
                return None
            
            self.service_stats["total_agent_requests"] += 1
            
            # Process the request
            response = await agent.process_request(user_input, context_data)
            
            if response.success:
                self.service_stats["successful_requests"] += 1
                self.logger.info(f"Request processed successfully by {agent_name}")
            else:
                self.service_stats["failed_requests"] += 1
                self.logger.warning(f"Request failed for {agent_name}: {response.error}")
            
            return response
            
        except Exception as e:
            self.service_stats["failed_requests"] += 1
            self.logger.error(f"Error processing request with {agent_name}: {e}")
            return None
    
    def start_session(self, session_id: str, agent_name: str, 
                     user_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start a new agent session.
        
        Args:
            session_id: Unique session identifier
            agent_name: Name of the agent to use
            user_info: Additional user information
            
        Returns:
            True if session started successfully, False otherwise
        """
        try:
            if session_id in self.active_sessions:
                self.logger.warning(f"Session {session_id} already exists")
                return False
            
            agent = self.get_agent(agent_name)
            if not agent:
                self.logger.error(f"Cannot start session: agent {agent_name} not found")
                return False
            
            session_info = {
                "session_id": session_id,
                "agent_name": agent_name,
                "start_time": datetime.now().isoformat(),
                "user_info": user_info or {},
                "interaction_count": 0,
                "last_activity": datetime.now().isoformat()
            }
            
            self.active_sessions[session_id] = session_info
            self.service_stats["sessions_started"] += 1
            
            self.logger.info(f"Started session {session_id} with agent {agent_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start session {session_id}: {e}")
            return False
    
    def end_session(self, session_id: str) -> bool:
        """End an active session."""
        try:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Session {session_id} not found")
                return False
            
            session_info = self.active_sessions.pop(session_id)
            
            # Clean up agent history if needed
            agent = self.get_agent(session_info["agent_name"])
            if agent:
                # Optionally clear history or save it
                pass
            
            self.logger.info(f"Ended session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to end session {session_id}: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.active_sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return list(self.active_sessions.values())
    
    async def get_agent_conversation_summary(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary for a specific agent."""
        try:
            agent = self.get_agent(agent_name)
            if not agent:
                return None
            
            # For echo agents, get conversation summary
            if hasattr(agent, 'get_conversation_summary'):
                return await agent.get_conversation_summary()
            else:
                # Fallback to basic stats
                stats = agent.get_stats()
                return {
                    "agent_name": agent_name,
                    "stats": stats,
                    "history_count": len(agent.get_history())
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get conversation summary for {agent_name}: {e}")
            return None
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get overall service statistics."""
        return {
            **self.service_stats,
            "active_agents": len(self.agents),
            "active_sessions": len(self.active_sessions),
            "success_rate": (
                self.service_stats["successful_requests"] / self.service_stats["total_agent_requests"]
                if self.service_stats["total_agent_requests"] > 0 else 0.0
            )
        }
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the service."""
        try:
            if name not in self.agents:
                self.logger.warning(f"Agent {name} not found")
                return False
            
            agent = self.agents.pop(name)
            agent.cleanup()
            
            self.logger.info(f"Removed agent: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove agent {name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up all agents and sessions."""
        try:
            # End all sessions
            for session_id in list(self.active_sessions.keys()):
                self.end_session(session_id)
            
            # Clean up all agents
            for name, agent in self.agents.items():
                try:
                    agent.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up agent {name}: {e}")
            
            self.agents.clear()
            self.logger.info("Agent Service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during service cleanup: {e}")


# Convenience functions for common operations
async def create_echo_agent_service() -> AgentService:
    """Create an agent service with a default echo agent."""
    service = AgentService()
    
    # Create default echo agent
    echo_agent = service.create_agent("echo", "Default Echo Agent")
    if echo_agent:
        service.logger.info("Default echo agent created successfully")
    
    return service


async def test_agent_service():
    """Test the agent service functionality."""
    print("Testing Agent Service...")
    print("=" * 50)
    
    service = await create_echo_agent_service()
    
    try:
        # List agents
        print("\n1. Available agents:")
        agents = service.list_agents()
        for agent in agents:
            print(f"   - {agent['name']} ({agent['type']})")
        
        # Start a session
        print("\n2. Starting session...")
        session_id = "test_session_001"
        if service.start_session(session_id, "Default Echo Agent"):
            print(f"   ✅ Session started: {session_id}")
        else:
            print("   ❌ Failed to start session")
        
        # Process a request
        print("\n3. Processing request...")
        response = await service.process_request("Default Echo Agent", "Hello, this is a test!")
        if response and response.success:
            print(f"   ✅ Response: {response.content}")
            print(f"   Processing time: {response.processing_time:.2f}s")
        else:
            print(f"   ❌ Request failed: {response.error if response else 'No response'}")
        
        # Get service stats
        print("\n4. Service statistics:")
        stats = service.get_service_stats()
        print(f"   Total requests: {stats['total_agent_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Active agents: {stats['active_agents']}")
        print(f"   Active sessions: {stats['active_sessions']}")
        
        # End session
        print("\n5. Ending session...")
        if service.end_session(session_id):
            print(f"   ✅ Session ended: {session_id}")
        else:
            print("   ❌ Failed to end session")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        service.cleanup()
        print("\n✅ Agent service test completed")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_agent_service())

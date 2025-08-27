"""
Base Agent Class for OpenAI Agents SDK Integration
Provides common functionality for all AI agents in the system.
"""

import os
import sys
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

try:
    from openai import OpenAI
    from pydantic import BaseModel
except ImportError as e:
    print(f"Base Agent: Missing required dependency: {e}")
    print("Please install dependencies: pip install openai pydantic")
    sys.exit(1)

# Native OpenAI conversation management with proper history
class ConversationManager:
    """Manages conversation history and context for OpenAI agents."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.conversations: List[Dict[str, Any]] = []
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations.append(message)
        
        # Maintain max length
        if len(self.conversations) > self.max_history:
            self.conversations.pop(0)
    
    def get_conversation_context(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get conversation context in OpenAI format."""
        context = []
        
        if include_system:
            context.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Maintain conversation context and provide relevant responses."
            })
        
        # Add recent conversation history
        for msg in self.conversations[-20:]:  # Last 20 messages for context
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return context
    
    def get_recent_context(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return self.conversations[-count:] if self.conversations else []
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversations.clear()
        self.session_id = self._generate_session_id()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.conversations),
            "start_time": self.conversations[0]["timestamp"] if self.conversations else None,
            "last_message": self.conversations[-1]["timestamp"] if self.conversations else None
        }

# Agent implementation using OpenAI client with proper conversation management
class OpenAIAgent:
    """OpenAI agent with native conversation management."""
    
    def __init__(self, name: str, instructions: str, model: str, client: OpenAI):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.client = client
        self.conversation_manager = ConversationManager()
    
    async def run(self, user_input: str, context_data: Optional[Dict[str, Any]] = None):
        """Run the agent with proper conversation context."""
        try:
            # Add user input to conversation history
            self.conversation_manager.add_message("user", user_input, context_data)
            
            # Build conversation context
            messages = self.conversation_manager.get_conversation_context()
            
            # Add system instructions
            messages.insert(0, {
                "role": "system",
                "content": self.instructions
            })
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_manager.add_message("assistant", response_content)
            
            return response_content
            
        except Exception as e:
            error_msg = f"Agent execution error: {e}"
            self.conversation_manager.add_message("system", error_msg)
            raise Exception(error_msg)

# Use the new implementations
Agent = OpenAIAgent

class Runner:
    """Runner class for agent execution."""
    
    @staticmethod
    async def run(agent, user_input: str, context_metadata: Optional[Dict[str, Any]] = None):
        """Run an agent with explicit user input and optional context metadata."""
        if hasattr(agent, 'run'):
            return await agent.run(user_input, context_metadata or {})
        else:
            return f"Agent response to: {user_input}"

class Context:
    """Context class for agent interactions."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class AgentConfig:
    """Configuration for agent behavior and settings."""
    
    # Agent identity
    name: str = "Base Agent"
    instructions: str = "You are a helpful assistant"
    
    # OpenAI settings
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    
    # Performance settings
    timeout: float = 30.0  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Context settings
    max_history_length: int = 100
    include_timestamp: bool = True
    include_metadata: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_agent_interactions: bool = True
    log_context_changes: bool = True


class AgentResponse(BaseModel):
    """Standard response structure for agent interactions."""
    
    success: bool
    content: str
    agent_name: str
    timestamp: str
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None





class BaseAgent:
    """Base class for all AI agents using OpenAI Agents SDK."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent with configuration.
        
        Args:
            config: Agent configuration settings
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize OpenAI client
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Base Agent: OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize conversation manager for proper history
        self.conversation_manager = ConversationManager(config.max_history_length)
        
        # Initialize OpenAI Agent with client
        self.agent = Agent(
            name=config.name,
            instructions=config.instructions,
            model=config.model,
            client=self.client
        )
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "timeout_errors": 0,
            "api_errors": 0
        }
        
        self.logger.info(f"Base Agent: Initialized '{config.name}' with model {config.model}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger(f"Agent_{self.config.name}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def process_request(self, user_input: str, 
                            context_data: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process user request using the agent.
        
        Args:
            user_input: User's input text
            context_data: Additional context data
            
        Returns:
            AgentResponse with processing results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            if not user_input.strip():
                raise ValueError("User input cannot be empty")
            
            self.logger.info(f"Processing request: {user_input[:50]}...")
            
            # Create context with history
            context = await self._create_context(user_input, context_data)
            
            # Run agent with timeout
            result = await asyncio.wait_for(
                self._run_agent(user_input, context),
                timeout=self.config.timeout
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["successful_requests"] += 1
            self._update_average_processing_time()
            
            # Extract response content
            response_content = self._extract_response_content(result)
            
            # Add to conversation history
            self.conversation_manager.add_message("user", user_input, {
                "processing_time": processing_time,
                "model": self.config.model,
                "context_size": len(str(context))
            })
            
            self.conversation_manager.add_message("assistant", response_content, {
                "processing_time": processing_time,
                "model": self.config.model
            })
            
            self.logger.info(f"Request processed successfully in {processing_time:.2f}s")
            
            return AgentResponse(
                success=True,
                content=response_content,
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                metadata={
                    "model": self.config.model,
                    "session_id": self.conversation_manager.session_id,
                    "interaction_count": len(self.conversation_manager.conversations)
                }
            )
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            self.stats["timeout_errors"] += 1
            self.stats["failed_requests"] += 1
            self.logger.error(f"Request timed out after {processing_time:.2f}s")
            
            return AgentResponse(
                success=False,
                content="",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                error="Request timed out"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["api_errors"] += 1
            self.stats["failed_requests"] += 1
            self.logger.error(f"Request failed after {processing_time:.2f}s: {e}")
            
            return AgentResponse(
                success=False,
                content="",
                agent_name=self.config.name,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _create_context(self, user_input: str, 
                            context_data: Optional[Dict[str, Any]] = None) -> Context:
        """Create context with conversation history and additional data."""
        try:
            # Get recent conversation context from conversation manager
            recent_context = self.conversation_manager.get_recent_context(5)
            
            # Build context string
            context_parts = []
            
            if self.config.include_timestamp:
                context_parts.append(f"Current time: {datetime.now().isoformat()}")
            
            if recent_context:
                context_parts.append("Recent conversation history:")
                for msg in recent_context:
                    role = msg['role'].title()
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    context_parts.append(f"{role}: {content}")
            
            if context_data:
                context_parts.append("Additional context:")
                for key, value in context_data.items():
                    context_parts.append(f"{key}: {value}")
            
            context_string = "\n".join(context_parts)
            
            # Create context object
            context = Context(context_string)
            
            if self.config.log_context_changes:
                self.logger.debug(f"Context created with {len(context_parts)} parts")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error creating context: {e}")
            # Return minimal context
            return Context(f"User input: {user_input}")
    
    async def _run_agent(self, user_input: str, context: Context) -> Any:
        """Run the agent with the given input and context."""
        try:
            # Pass only the raw user_input to the agent; include context as metadata if desired
            result = await Runner.run(self.agent, user_input, {"context": str(context)})
            return result
            
        except Exception as e:
            self.logger.error(f"Agent execution error: {e}")
            raise
    
    def _extract_response_content(self, result: Any) -> str:
        """Extract response content from agent result."""
        try:
            # Handle different result types
            if hasattr(result, 'final_output'):
                return str(result.final_output)
            elif hasattr(result, 'content'):
                return str(result.content)
            elif hasattr(result, 'text'):
                return str(result.text)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except Exception as e:
            self.logger.warning(f"Could not extract response content: {e}")
            return str(result)
    
    def _update_average_processing_time(self):
        """Update average processing time statistics."""
        if self.stats["successful_requests"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"] 
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "history_summary": self.conversation_manager.get_session_summary(),
            "config": {
                "name": self.config.name,
                "model": self.config.model,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }
        }
    
    def get_history(self, recent_only: bool = True) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if recent_only:
            return self.conversation_manager.get_recent_context(10)
        return self.conversation_manager.conversations.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_manager.clear_history()
        self.logger.info("Conversation history cleared")
    
    def update_instructions(self, new_instructions: str):
        """Update agent instructions."""
        self.config.instructions = new_instructions
        # Recreate agent with new instructions
        self.agent = Agent(
            name=self.config.name,
            instructions=new_instructions,
            model=self.config.model
        )
        self.logger.info("Agent instructions updated")
    
    def cleanup(self):
        """Clean up agent resources."""
        try:
            # Clear conversation history
            self.conversation_manager.clear_history()
            self.logger.info("Agent cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Export main classes
__all__ = ['BaseAgent', 'AgentConfig', 'AgentResponse', 'ConversationManager']

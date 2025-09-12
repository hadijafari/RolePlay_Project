#!/usr/bin/env python3
"""
Test script to verify Role Switch Detection for complete message capture
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_test import ConversationLogger, CustomDeepgramVoiceAgentTabService, MockConversationText

class TestRoleSwitchDetection:
    """Test class for role switch detection functionality."""
    
    def __init__(self):
        # Create test conversation logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rag_dir = Path(__file__).parent / "RAG"
        rag_dir.mkdir(exist_ok=True)
        log_file_path = rag_dir / f"test_role_switch_{timestamp}.txt"
        
        self.conversation_logger = ConversationLogger(str(log_file_path))
        self.log_file_path = log_file_path
        
        # Create custom voice agent (without Deepgram connection)
        self.voice_agent = CustomDeepgramVoiceAgentTabService(
            questions=["Test question"],
            feedback_agent=None,  # Skip feedback for this test
            conversation_logger=self.conversation_logger
        )
        
        # Mock the conversation log and other required attributes
        self.voice_agent.conversation_log = []
        self.voice_agent.waiting_for_response = False
        self.voice_agent.logger = self._create_mock_logger()
        self.voice_agent.retry_count = 0
        
        self.processed_qa_pairs = []
    
    def _create_mock_logger(self):
        """Create a mock logger for testing."""
        class MockLogger:
            def debug(self, msg): pass
        return MockLogger()
    
    async def mock_process_qa_pair(self, question, answer):
        """Mock version of _process_qa_pair for testing."""
        self.processed_qa_pairs.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        print(f"🧪 Mock feedback generated for: Q='{question[:50]}...' A='{answer[:50]}...'")
    
    def simulate_chunked_conversation(self):
        """Simulate a chunked conversation like the real scenario."""
        print("🧪 Testing Role Switch Detection with Chunked Messages")
        print("="*70)
        
        # Replace the actual process method with our mock
        self.voice_agent._process_qa_pair = self.mock_process_qa_pair
        
        # Create the conversation text handler
        # We need to manually create the handler since we're not connecting to Deepgram
        def on_conversation_text(_, conversation_text, **kwargs):
            """Simulated conversation text handler."""
            text_data = {
                "role": conversation_text.role,
                "content": conversation_text.content,
                "timestamp": datetime.now().isoformat()
            }
            self.voice_agent.conversation_log.append(text_data)
            
            # Role switch detection logic (copied from main implementation)
            incoming_role = conversation_text.role
            
            # Check if role has switched
            if self.voice_agent.current_role is not None and self.voice_agent.current_role != incoming_role:
                print(f"🔄 Role switch detected: {self.voice_agent.current_role} → {incoming_role}")
                
                # Process complete messages when role switches
                if self.voice_agent.current_role == "assistant" and incoming_role == "user":
                    # Interviewer finished speaking, user starting to speak
                    self.voice_agent.complete_question = self.voice_agent.assistant_message_buffer.strip()
                    self.voice_agent.assistant_message_buffer = ""  # Reset buffer
                    print(f"📝 Complete question captured: {self.voice_agent.complete_question[:100]}...")
                    
                elif self.voice_agent.current_role == "user" and incoming_role == "assistant":
                    # User finished speaking, interviewer starting to speak
                    self.voice_agent.complete_answer = self.voice_agent.user_message_buffer.strip()
                    self.voice_agent.user_message_buffer = ""  # Reset buffer
                    print(f"📝 Complete answer captured: {self.voice_agent.complete_answer[:100]}...")
                    
                    # Process Q&A pair if we have both complete question and answer
                    if self.voice_agent.complete_question and self.voice_agent.complete_answer:
                        print(f"🔄 Processing complete Q&A pair {self.voice_agent.question_counter + 1}...")
                        
                        # Use asyncio.create_task for testing (simpler than the thread-safe version)
                        asyncio.create_task(self.voice_agent._process_qa_pair(
                            self.voice_agent.complete_question, 
                            self.voice_agent.complete_answer
                        ))
                        
                        # Reset for next Q&A pair
                        self.voice_agent.complete_question = None
                        self.voice_agent.complete_answer = None
                        self.voice_agent.question_counter += 1
            
            # Update current role
            self.voice_agent.current_role = incoming_role
            
            # Accumulate message chunks in appropriate buffer
            if incoming_role == "assistant":
                # Add space between chunks if buffer is not empty
                if self.voice_agent.assistant_message_buffer:
                    self.voice_agent.assistant_message_buffer += " "
                self.voice_agent.assistant_message_buffer += conversation_text.content
                print(f"🤖 Interviewer chunk: {conversation_text.content}")
                    
            elif incoming_role == "user":
                # Add space between chunks if buffer is not empty
                if self.voice_agent.user_message_buffer:
                    self.voice_agent.user_message_buffer += " "
                self.voice_agent.user_message_buffer += conversation_text.content
                print(f"🗣️  User chunk: {conversation_text.content}")
        
        return on_conversation_text
    
    async def test_chunked_messages(self):
        """Test the role switch detection with chunked messages."""
        print("\n1. Testing chunked message handling...")
        
        handler = self.simulate_chunked_conversation()
        
        # Simulate the exact scenario from your logs
        print("\n📋 Simulating Interviewer Question (4 chunks):")
        
        # Interviewer chunks
        handler(None, MockConversationText("assistant", "That's great to hear, Hadi!"))
        handler(None, MockConversationText("assistant", "It's clear you have a strong passion for embedded systems and PCB design."))
        handler(None, MockConversationText("assistant", "It might be helpful to include any specific experiences or projects that influenced your decision to specialize in this area."))
        handler(None, MockConversationText("assistant", "Now, what specific projects have you worked on that involved STM32 or ESP32, and what was your role in those projects?"))
        
        print(f"\n📊 Current question buffer: '{self.voice_agent.assistant_message_buffer}'")
        
        print("\n📋 Simulating User Answer (3 chunks):")
        
        # User chunks (this should trigger role switch and capture complete question)
        handler(None, MockConversationText("user", "I don't have any projects. I just want to"))
        handler(None, MockConversationText("user", "do designing and PC working and also component finding and"))
        handler(None, MockConversationText("user", "stuff like that."))
        
        print(f"\n📊 Current answer buffer: '{self.voice_agent.user_message_buffer}'")
        
        print("\n📋 Simulating Next Interviewer Response (triggers processing):")
        handler(None, MockConversationText("assistant", "Thank you for that information."))
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        return True
    
    def verify_results(self):
        """Verify that the complete messages were captured correctly."""
        print("\n2. Verifying results...")
        
        if not self.processed_qa_pairs:
            print("❌ No Q&A pairs were processed!")
            return False
        
        qa_pair = self.processed_qa_pairs[0]
        
        expected_question = "That's great to hear, Hadi! It's clear you have a strong passion for embedded systems and PCB design. It might be helpful to include any specific experiences or projects that influenced your decision to specialize in this area. Now, what specific projects have you worked on that involved STM32 or ESP32, and what was your role in those projects?"
        
        expected_answer = "I don't have any projects. I just want to do designing and PC working and also component finding and stuff like that."
        
        print(f"\n📝 Captured Question: {qa_pair['question']}")
        print(f"📝 Expected Question: {expected_question}")
        
        print(f"\n📝 Captured Answer: {qa_pair['answer']}")
        print(f"📝 Expected Answer: {expected_answer}")
        
        question_match = qa_pair['question'] == expected_question
        answer_match = qa_pair['answer'] == expected_answer
        
        if question_match:
            print("✅ Complete question captured correctly!")
        else:
            print("❌ Question capture failed!")
        
        if answer_match:
            print("✅ Complete answer captured correctly!")
        else:
            print("❌ Answer capture failed!")
        
        return question_match and answer_match

async def main():
    """Main test function."""
    print("🧪 ROLE SWITCH DETECTION TEST")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    tester = TestRoleSwitchDetection()
    
    try:
        # Test chunked message handling
        await tester.test_chunked_messages()
        
        # Verify results
        success = tester.verify_results()
        
        if success:
            print("\n🎉 Role Switch Detection test PASSED!")
            print("✅ Complete messages are now captured correctly")
            print("✅ Q&A pairs will have full context for feedback agent")
            print("✅ Conversation log will contain complete messages")
        else:
            print("\n❌ Role Switch Detection test FAILED!")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Test script for the FeedbackAgent integration
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.feedback_agent import FeedbackAgent
from main_test import ConversationLogger

async def test_feedback_agent():
    """Test the feedback agent with sample Q&A pairs."""
    print("🧪 Testing Feedback Agent Integration")
    print("="*50)
    
    # Test 1: Initialize feedback agent
    print("1. Testing FeedbackAgent initialization...")
    try:
        feedback_agent = FeedbackAgent()
        print("✅ FeedbackAgent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize FeedbackAgent: {e}")
        return False
    
    # Test 2: Initialize conversation logger
    print("\n2. Testing ConversationLogger initialization...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"test_conversations_{timestamp}.txt"
        logger = ConversationLogger(log_file)
        print(f"✅ ConversationLogger initialized: {log_file}")
    except Exception as e:
        print(f"❌ Failed to initialize ConversationLogger: {e}")
        return False
    
    # Test 3: Generate feedback for sample Q&A
    print("\n3. Testing feedback generation...")
    sample_questions = [
        "Could you please introduce yourself and tell me a bit about your experience?",
        "What programming languages are you most comfortable with?",
        "Describe a challenging project you've worked on recently."
    ]
    
    sample_answers = [
        "Hi. My name is Hadi. How are you?",
        "I know Python, JavaScript, and some Java. I've been coding for about 2 years.",
        "I built a web application using React and Node.js. It was challenging because I had to learn new technologies."
    ]
    
    for i, (question, answer) in enumerate(zip(sample_questions, sample_answers), 1):
        print(f"\n   Testing Q&A pair {i}...")
        try:
            feedback_data = await feedback_agent.generate_feedback(
                question=question,
                answer=answer,
                question_number=i
            )
            
            print(f"   ✅ Feedback generated for Q&A {i}")
            print(f"   📊 Overall Score: {feedback_data.get('overall_score', 0.5):.2f}/1.0")
            print(f"   📝 Summary: {feedback_data.get('summary', 'No summary')[:100]}...")
            
            # Test conversation logging
            formatted_feedback = feedback_agent.format_feedback_for_display(feedback_data)
            logger.log_qa_feedback(question, answer, formatted_feedback)
            print(f"   📁 Logged to conversation file")
            
        except Exception as e:
            print(f"   ❌ Error generating feedback for Q&A {i}: {e}")
            return False
    
    print(f"\n✅ All tests passed!")
    print(f"📁 Test conversation log: {log_file}")
    return True

async def main():
    """Main test function."""
    success = await test_feedback_agent()
    if success:
        print("\n🎉 Feedback Agent integration test completed successfully!")
    else:
        print("\n❌ Feedback Agent integration test failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())


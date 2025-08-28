"""
Test Interview Integration
Quick test to verify that the Interviewer Agent properly uses interview plan data.
"""

import asyncio
import json
from pathlib import Path
from agents.test_agent import InterviewerAgent


async def test_interview_with_plan():
    """Test the Interviewer Agent with actual interview plan data."""
    print("🧪 Testing Interview Agent Integration")
    print("=" * 60)
    
    # Load the existing interview plan
    plan_file = Path("RAG/interview_plan.json")
    
    if not plan_file.exists():
        print("❌ No interview plan found. Run the main program first.")
        return
    
    with open(plan_file, 'r', encoding='utf-8') as f:
        interview_plan = json.load(f)
    
    print(f"✅ Loaded interview plan from {plan_file}")
    print(f"   Analysis timestamp: {interview_plan.get('analysis_timestamp', 'Unknown')}")
    
    # Create interviewer agent
    interviewer = InterviewerAgent()
    
    # Test with interview plan context
    test_messages = [
        "Hello, I'm ready to start the interview.",
        "Tell me about my background.",
        "What technical questions do you have for me?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Testing: '{message}'")
        print("-" * 40)
        
        # Create context with interview plan
        context_data = {
            "interview_plan": interview_plan,
            "context": "Testing interview plan integration"
        }
        
        try:
            response = await interviewer.process_interview_message(message, context_data)
            
            if response.success:
                print(f"✅ Agent Response:")
                print(f"   {response.content}")
                print(f"   (Processing time: {response.processing_time:.2f}s)")
                
                # Check if response mentions specific details from resume
                content_lower = response.content.lower()
                resume_indicators = [
                    "mohammadhadi", "embedded", "pcb", "stm32", "esp32", 
                    "center of infusion", "artin system", "gui development",
                    "visual c++", "medical device"
                ]
                
                found_indicators = [ind for ind in resume_indicators if ind in content_lower]
                
                if found_indicators:
                    print(f"   🎯 References resume/job data: {', '.join(found_indicators)}")
                else:
                    print(f"   ⚠️  No specific resume/job references detected")
                    
            else:
                print(f"❌ Agent failed: {response.error}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed!")


async def test_interview_without_plan():
    """Test the Interviewer Agent without interview plan (fallback behavior)."""
    print("\n🧪 Testing Interview Agent Without Plan (Fallback)")
    print("=" * 60)
    
    interviewer = InterviewerAgent()
    
    message = "Hello, I'm ready for the interview."
    
    try:
        response = await interviewer.process_interview_message(message, {})
        
        if response.success:
            print(f"✅ Fallback Response: {response.content}")
        else:
            print(f"❌ Fallback failed: {response.error}")
            
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")


async def main():
    """Run integration tests."""
    try:
        await test_interview_with_plan()
        await test_interview_without_plan()
        
        print("\n🎉 All integration tests completed!")
        
    except Exception as e:
        print(f"\n❌ Integration test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

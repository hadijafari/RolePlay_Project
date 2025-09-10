#!/usr/bin/env python3
"""
Simplified test file for Deepgram Voice Agent with hardcoded questions.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.deepgram_voice_agent_tab_service import DeepgramVoiceAgentTabService


async def main():
    """Main test function."""
    print("🧪 SIMPLIFIED DEEPGRAM VOICE AGENT TEST")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Initialize simplified voice agent (no complex objects needed)
    voice_agent = DeepgramVoiceAgentTabService()
    
    try:
        # Connect to Deepgram Voice Agent
        if await voice_agent.connect():
            print("✅ Connected to Deepgram Voice Agent")
            print("\n" + "="*60)
            print("🎤 INTERVIEW READY")
            print("📝 Hold TAB to speak, release to send")
            print("❌ Press Ctrl+C to exit")
            print("="*60)
            
            # Start the interview loop
            await voice_agent.start_interview()
            
        else:
            print("❌ Failed to connect to Deepgram Voice Agent")
            
    except KeyboardInterrupt:
        print("\n🛑 Interview stopped by user")
    except Exception as e:
        print(f"❌ Error during interview: {e}")
    finally:
        await voice_agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

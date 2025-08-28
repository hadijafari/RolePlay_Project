#!/usr/bin/env python3
"""
Test Script for Interview Platform System
Tests the complete flow: document analysis -> interview plan -> interview agent
"""

import asyncio
import json
import os
from pathlib import Path

def check_documents():
    """Check if required documents exist in the RAG directory."""
    print("🔍 Checking for required documents...")
    
    rag_dir = Path("RAG")
    if not rag_dir.exists():
        print("❌ RAG directory not found")
        print("   Creating RAG directory...")
        rag_dir.mkdir(exist_ok=True)
    
    # Look for resume files
    resume_files = []
    for ext in ['.pdf', '.docx', '.doc', '.txt']:
        resume_files.extend(rag_dir.glob(f"resume{ext}"))
        resume_files.extend(rag_dir.glob(f"resume*{ext}"))
        resume_files.extend(rag_dir.glob(f"cv{ext}"))
        resume_files.extend(rag_dir.glob(f"cv*{ext}"))
    
    # Look for job description files
    job_files = []
    for ext in ['.pdf', '.docx', '.doc', '.txt']:
        job_files.extend(rag_dir.glob(f"description{ext}"))
        job_files.extend(rag_dir.glob(f"description*{ext}"))
        job_files.extend(rag_dir.glob(f"job{ext}"))
        job_files.extend(rag_dir.glob(f"job*{ext}"))
    
    print(f"📄 Resume files found: {[f.name for f in resume_files]}")
    print(f"💼 Job description files found: {[f.name for f in job_files]}")
    
    if not resume_files:
        print("⚠️  No resume found. Please place your resume in the RAG/ folder as:")
        print("   - resume.pdf, resume.docx, resume.txt, or")
        print("   - cv.pdf, cv.docx, cv.txt")
    
    if not job_files:
        print("⚠️  No job description found. Please place the job description in the RAG/ folder as:")
        print("   - description.pdf, description.docx, description.txt, or")
        print("   - job.pdf, job.docx, job.txt")
    
    return len(resume_files) > 0 and len(job_files) > 0

async def test_document_intelligence():
    """Test the document intelligence system."""
    print("\n🤖 Testing Document Intelligence System...")
    
    try:
        from agents.document_intelligence_agent import DocumentIntelligenceAgent
        
        # Create the agent
        doc_agent = DocumentIntelligenceAgent(
            name="Document Intelligence Agent",
            instructions="Analyze the provided resume and job description to create a comprehensive interview strategy.",
            model="gpt-4o-mini"
        )
        
        print("✅ Document Intelligence Agent created")
        
        # Check if interview plan already exists
        existing_plan = await doc_agent.get_interview_plan()
        if existing_plan:
            print("📋 Existing interview plan found!")
            return existing_plan
        
        # Run document analysis
        print("📄 Running document analysis...")
        analysis_result = await doc_agent.analyze_documents()
        
        if analysis_result and analysis_result.get("success"):
            print("✅ Document analysis completed successfully!")
            return analysis_result
        else:
            error_msg = analysis_result.get('error', 'Unknown error') if analysis_result else 'No response'
            print(f"❌ Document analysis failed: {error_msg}")
            return None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def test_interview_agent(interview_plan):
    """Test the interview agent with the generated plan."""
    print("\n🎤 Testing Interview Agent...")
    
    try:
        from agents.test_agent import InterviewerAgent
        
        # Create the interview agent
        interview_agent = InterviewerAgent(
            name="Interview Assistant",
            instructions="Conduct an intelligent interview based on the provided interview plan.",
            model="gpt-4o-mini"
        )
        
        print("✅ Interview Agent created")
        
        # Test with a sample question
        test_question = "Hello, I'm ready to start the interview."
        context_data = {
            "interview_plan": interview_plan,
            "source": "test"
        }
        
        print(f"🤔 Test question: {test_question}")
        
        # Process the question
        if hasattr(interview_agent, 'process_interview_message'):
            response = await interview_agent.process_interview_message(test_question, context_data)
        else:
            response = await interview_agent.process_request(test_question, context_data)
        
        if response and response.success:
            print(f"✅ Agent response: {response.content[:200]}...")
            return True
        else:
            error_msg = response.error if response else "No response"
            print(f"❌ Agent processing failed: {error_msg}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Interview Platform System Test")
    print("=" * 60)
    
    # Check documents
    documents_ready = check_documents()
    
    if not documents_ready:
        print("\n❌ Cannot proceed without required documents.")
        print("   Please add your resume and job description to the RAG/ folder and run again.")
        return
    
    print("\n✅ Documents found. Proceeding with tests...")
    
    # Test document intelligence
    interview_plan = asyncio.run(test_document_intelligence())
    
    if not interview_plan:
        print("\n❌ Document intelligence test failed.")
        return
    
    print("\n✅ Document intelligence test passed!")
    
    # Test interview agent
    agent_success = asyncio.run(test_interview_agent(interview_plan))
    
    if agent_success:
        print("\n✅ Interview agent test passed!")
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n🚀 You can now run the main program:")
        print("   uv run main.py")
    else:
        print("\n❌ Interview agent test failed.")
        print("   The system may not work properly.")

if __name__ == "__main__":
    main()

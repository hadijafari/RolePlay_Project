"""
Test Script for Document Intelligence Agent
Demonstrates the complete document analysis workflow.
"""

import asyncio
import json
from pathlib import Path

from agents.document_intelligence_agent import DocumentIntelligenceAgent
from services.agent_service import AgentService


def display_interview_plan_summary(analysis_result):
    """Display a nice summary of the interview plan in the terminal."""
    try:
        # Show candidate summary
        if "candidate_analysis" in analysis_result:
            candidate = analysis_result["candidate_analysis"]
            if "raw_analysis" in candidate:
                try:
                    import json
                    candidate_data = json.loads(candidate["raw_analysis"].replace("```json\n", "").replace("```", ""))
                    if "basic_information" in candidate_data:
                        info = candidate_data["basic_information"]
                        print(f"👤 CANDIDATE: {info.get('name', 'N/A')}")
                        if "contact" in info:
                            print(f"   📧 Email: {info['contact'].get('email', 'N/A')}")
                            print(f"   📱 Phone: {info['contact'].get('phone', 'N/A')}")
                except:
                    print("👤 CANDIDATE: Analysis completed")
        
        # Show job summary
        if "job_analysis" in analysis_result:
            job = analysis_result["job_analysis"]
            if "raw_analysis" in job:
                try:
                    job_data = json.loads(job["raw_analysis"].replace("```json\n", "").replace("```", ""))
                    print(f"💼 JOB: {job_data.get('job_title', 'N/A')}")
                    print(f"   🏢 Industry: {job_data.get('industry', 'N/A')}")
                    print(f"   📊 Level: {job_data.get('seniority_level', 'N/A')}")
                except:
                    print("💼 JOB: Analysis completed")
        
        # Show gap analysis
        if "gap_analysis" in analysis_result:
            gap = analysis_result["gap_analysis"]
            if "raw_analysis" in gap:
                try:
                    gap_data = json.loads(gap["raw_analysis"].replace("```json\n", "").replace("```", ""))
                    print(f"🔍 GAP ANALYSIS:")
                    print(f"   📊 Overall Fit: {gap_data.get('overall_fit_score', 'N/A')}%")
                    print(f"   📊 Skills Match: {gap_data.get('skills_match_percentage', 'N/A')}%")
                    
                    if "missing_critical_skills" in gap_data and gap_data["missing_critical_skills"]:
                        print(f"   ❌ Missing Critical Skills: {', '.join(gap_data['missing_critical_skills'][:3])}")
                    
                    if "opportunity_areas" in gap_data and gap_data["opportunity_areas"]:
                        print(f"   ✅ Opportunity Areas: {gap_data['opportunity_areas'][0]}")
                except:
                    print("🔍 GAP ANALYSIS: Completed")
        
        # Show interview strategy highlights
        if "interview_strategy" in analysis_result:
            strategy = analysis_result["interview_strategy"]
            if "raw_analysis" in strategy:
                try:
                    strategy_data = json.loads(strategy["raw_analysis"].replace("```json\n", "").replace("```", ""))
                    print(f"📋 INTERVIEW STRATEGY:")
                    
                    if "technical_questions" in strategy_data:
                        tech_areas = list(strategy_data["technical_questions"].keys())
                        print(f"   🔧 Technical Areas: {', '.join(tech_areas[:3])}")
                    
                    if "gap_analysis_questions" in strategy_data:
                        gap_areas = list(strategy_data["gap_analysis_questions"].keys())
                        print(f"   🎯 Gap Focus: {', '.join(gap_areas[:3])}")
                    
                    if "interview_flow_and_focus_areas" in strategy_data:
                        flow = strategy_data["interview_flow_and_focus_areas"]
                        print(f"   📝 Interview Flow: {len(flow)} focus areas defined")
                except:
                    print("📋 INTERVIEW STRATEGY: Generated")
                    
    except Exception as e:
        print(f"⚠️  Error displaying plan summary: {e}")


async def test_document_intelligence():
    """Test the complete document intelligence workflow."""
    print("🤖 Document Intelligence Agent Test")
    print("=" * 60)
    
    try:
        # Initialize the agent service
        print("1. Initializing Agent Service...")
        agent_service = AgentService()
        
        # Create the Document Intelligence Agent
        print("2. Creating Document Intelligence Agent...")
        doc_agent = agent_service.create_agent(
            "document_intelligence", 
            "Document Analysis Agent",
            model="gpt-4o"
        )
        
        if not doc_agent:
            print("❌ Failed to create Document Intelligence Agent")
            return
        
        print("✅ Document Intelligence Agent created successfully")
        
        # Check RAG directory
        rag_dir = doc_agent.get_rag_directory()
        print(f"3. RAG Directory: {rag_dir}")
        
        # List available documents
        documents = doc_agent.list_available_documents()
        print(f"4. Available Documents: {documents}")
        
        if not documents:
            print("⚠️  No documents found in RAG directory")
            print("   Please add resume and job description files to:")
            print(f"   {rag_dir}")
            print("   Expected files: resume.pdf/docx/txt, description.pdf/docx/txt")
            return
        
        # Check if interview plan already exists
        existing_plan = await doc_agent.get_interview_plan()
        if existing_plan:
            print("5. Found existing interview plan, loading...")
            print(f"   Timestamp: {existing_plan.get('analysis_timestamp', 'Unknown')}")
            print("   ✅ Interview plan is ready for use")
            
            # Display the existing interview plan
            print("\n" + "="*80)
            print("📋 EXISTING INTERVIEW PLAN SUMMARY")
            print("="*80)
            display_interview_plan_summary(existing_plan)
            print("="*80)
            return
        
        # Analyze documents
        print("5. Starting document analysis...")
        print("   This may take a few minutes...")
        
        analysis_result = await doc_agent.analyze_documents()
        
        if analysis_result["success"]:
            print("✅ Document analysis completed successfully!")
            print(f"   Analysis timestamp: {analysis_result['analysis_timestamp']}")
            
            # Show summary
            if "candidate_analysis" in analysis_result:
                print("   📄 Resume analyzed")
            if "job_analysis" in analysis_result:
                print("   💼 Job description analyzed")
            if "gap_analysis" in analysis_result:
                print("   🔍 Gap analysis completed")
            if "interview_strategy" in analysis_result:
                print("   📋 Interview strategy generated")
            
            print("\n🎯 Interview plan is ready!")
            print(f"   Saved to: {rag_dir}/interview_plan.json")
            
            # Display the interview plan in a nice format
            print("\n" + "="*80)
            print("📋 INTERVIEW PLAN SUMMARY")
            print("="*80)
            display_interview_plan_summary(analysis_result)
            print("="*80)
            
        else:
            print(f"❌ Document analysis failed: {analysis_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_interviewer_with_plan():
    """Test the Interviewer Agent with the generated plan."""
    print("\n🤖 Testing Interviewer Agent with Plan")
    print("=" * 60)
    
    try:
        # Initialize the agent service
        agent_service = AgentService()
        
        # Create the Document Intelligence Agent to access the plan
        doc_agent = agent_service.create_agent(
            "document_intelligence", 
            "Document Analysis Agent"
        )
        
        # Get the interview plan
        interview_plan = await doc_agent.get_interview_plan()
        
        if not interview_plan:
            print("❌ No interview plan found. Run document analysis first.")
            return
        
        print("✅ Interview plan loaded successfully")
        
        # Create the Interviewer Agent
        interviewer = agent_service.create_agent(
            "echo", 
            "Interview Assistant"
        )
        
        if not interviewer:
            print("❌ Failed to create Interviewer Agent")
            return
        
        print("✅ Interviewer Agent created successfully")
        
        # Test a sample interview question
        print("\n📝 Testing interview question...")
        test_question = "Hello, I'm ready to start the interview. What would you like to know about me?"
        
        response = await interviewer.process_request(test_question, {
            "interview_plan": interview_plan,
            "context": "Starting technical interview"
        })
        
        if response.success:
            print(f"✅ Interviewer Response: {response.content}")
        else:
            print(f"❌ Interviewer failed: {response.error}")
        
    except Exception as e:
        print(f"❌ Interviewer test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("🚀 Document Intelligence System Test Suite")
    print("=" * 60)
    
    try:
        # Test document analysis
        await test_document_intelligence()
        
        # Test interviewer with plan
        await test_interviewer_with_plan()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async test suite
    asyncio.run(main())

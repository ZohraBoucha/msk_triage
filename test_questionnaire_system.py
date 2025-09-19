#!/usr/bin/env python3
"""
Test script for the updated questionnaire-based MSK triage system.
This script tests the questionnaire engine and integration without requiring the full web interface.
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import with absolute imports
from app.questionnaire_engine import run_questionnaire_engine, get_diagnosis_display_name
from app.questionnaire_specs import get_questionnaire_form, KNEE_OA_SPEC, KNEE_INJURY_SPEC
from app.triage_agent import TriageAgent
from app.summarization_agent import SummarizationAgent

def test_questionnaire_engine():
    """Test the questionnaire engine with sample data."""
    print("🧪 Testing Questionnaire Engine...")
    
    # Test knee OA scenario
    print("\n📋 Testing Knee OA Assessment:")
    knee_oa_data = {
        "patient": {"age_years": 65},
        "laterality": "right",
        "duration_class": "chronic",
        "mechanism": "overuse",
        "phenotype": ["anterior_pain", "stiffness"],
        "oa_index": {
            "global_pain": "moderate",
            "stiffness_morning": "moderate",
            "stiffness_after_rest": "moderate",
            "function": {
                "stairs_down": "moderate",
                "stairs_up": "severe",
                "rise_from_sit": "moderate",
                "bend_to_floor": "mild",
                "in_out_car": "moderate",
                "socks_on_off": "moderate",
                "in_out_bath": "mild",
                "heavy_domestic": "severe"
            }
        },
        "exam": {
            "alignment": "varus",
            "effusion": "mild",
            "rom_restriction": True,
            "pf_crepitus": "bone"
        },
        "imaging": {"xray_oa_tf": True, "malalignment": "varus"},
        "red_flags": {
            "fever_unwell_hot_joint": False,
            "true_locked_knee": False,
            "inability_slr_after_eccentric_load": False
        }
    }
    
    result = run_questionnaire_engine(KNEE_OA_SPEC, knee_oa_data)
    print(f"Route: {result.get('route', 'Unknown')}")
    if result.get('top'):
        print("Top Diagnoses:")
        for i, dx in enumerate(result['top'][:3], 1):
            diagnosis_name = get_diagnosis_display_name(dx['diagnosis_code'])
            print(f"  {i}. {diagnosis_name} (Score: {dx['score']}, Confidence: {dx['confidence_band']})")
    
    # Test knee injury scenario
    print("\n🏥 Testing Knee Injury Assessment:")
    knee_injury_data = {
        "patient": {"age_years": 28},
        "laterality": "left",
        "duration_class": "acute",
        "mechanism": "twisting",
        "phenotype": ["instability", "locking_catching"],
        "knee_score": {
            "limp": 3,
            "pain": 15,
            "support": 2,
            "swelling": 6,
            "locking": 10,
            "stair_climbing": 6,
            "instability": 15,
            "squatting": 2
        },
        "exam": {
            "effusion": "moderate",
            "alignment": "normal",
            "lachman": "yes_soft_endpoint",
            "pivot_shift": True,
            "mcmurray_medial": False,
            "mcmurray_lateral": False
        },
        "imaging": {"xray_oa_tf": False, "malalignment": "none"},
        "red_flags": {
            "fever_unwell_hot_joint": False,
            "true_locked_knee": False,
            "inability_slr_after_eccentric_load": False
        }
    }
    
    result = run_questionnaire_engine(KNEE_INJURY_SPEC, knee_injury_data)
    print(f"Route: {result.get('route', 'Unknown')}")
    if result.get('top'):
        print("Top Diagnoses:")
        for i, dx in enumerate(result['top'][:3], 1):
            diagnosis_name = get_diagnosis_display_name(dx['diagnosis_code'])
            print(f"  {i}. {diagnosis_name} (Score: {dx['score']}, Confidence: {dx['confidence_band']})")

def test_triage_agent():
    """Test the triage agent with sample conversation."""
    print("\n🤖 Testing Triage Agent...")
    
    agent = TriageAgent()
    
    # Simulate a conversation
    messages = [
        {"role": "user", "content": "I have knee pain"},
        {"role": "assistant", "content": "I understand you're experiencing knee pain. Which knee is affected - left or right?"},
        {"role": "user", "content": "My right knee"},
        {"role": "assistant", "content": "Thank you. How long have you had this pain in your right knee?"},
        {"role": "user", "content": "About 6 months, it's been getting worse gradually"},
        {"role": "assistant", "content": "I see. How did this knee pain start - was it from an injury or did it come on gradually?"},
        {"role": "user", "content": "It came on gradually, no specific injury"},
        {"role": "assistant", "content": "Thank you. Can you describe what the pain feels like?"},
        {"role": "user", "content": "It's a dull, aching pain in the front of my knee"},
        {"role": "assistant", "content": "Does the pain spread anywhere else?"},
        {"role": "user", "content": "No, it stays in the front of my knee"},
        {"role": "assistant", "content": "Do you have any other symptoms like swelling, stiffness, or weakness?"},
        {"role": "user", "content": "Yes, I have morning stiffness that lasts about 20 minutes"},
        {"role": "assistant", "content": "Is the pain constant or does it come and go?"},
        {"role": "user", "content": "It's worse in the morning and after sitting for a while"},
        {"role": "assistant", "content": "What makes the pain better or worse?"},
        {"role": "user", "content": "Walking helps, but stairs and squatting make it worse"},
        {"role": "assistant", "content": "On a scale of 0 to 10, how severe is your pain?"},
        {"role": "user", "content": "About a 6 out of 10"},
        {"role": "assistant", "content": "How long does the morning stiffness last?"},
        {"role": "user", "content": "About 20-30 minutes"},
        {"role": "assistant", "content": "How is this affecting your daily activities?"},
        {"role": "user", "content": "I have trouble with stairs and getting up from chairs"},
        {"role": "assistant", "content": "Have you tried any treatments before?"},
        {"role": "user", "content": "I've tried paracetamol and some exercises from YouTube"},
        {"role": "assistant", "content": "To make sure we're not missing anything serious, have you experienced any fever, chills, unexplained weight loss, or severe weakness?"},
        {"role": "user", "content": "No, none of those"}
    ]
    
    # Test state determination
    current_state = agent._determine_current_state(messages)
    print(f"Current state: {current_state}")
    
    # Test patient data extraction
    patient_data = agent._extract_patient_data(messages)
    print(f"Extracted patient data: {patient_data}")

async def test_summarization_agent():
    """Test the summarization agent."""
    print("\n📝 Testing Summarization Agent...")
    
    agent = SummarizationAgent()
    
    # Sample conversation
    messages = [
        {"role": "user", "content": "I have knee pain"},
        {"role": "assistant", "content": "I understand you're experiencing knee pain. Which knee is affected - left or right?"},
        {"role": "user", "content": "My right knee, I'm 65 years old"},
        {"role": "assistant", "content": "Thank you. How long have you had this pain in your right knee?"},
        {"role": "user", "content": "About 6 months, it's been getting worse gradually"},
        {"role": "assistant", "content": "I see. How did this knee pain start - was it from an injury or did it come on gradually?"},
        {"role": "user", "content": "It came on gradually, no specific injury"},
        {"role": "assistant", "content": "Thank you. Can you describe what the pain feels like?"},
        {"role": "user", "content": "It's a dull, aching pain in the front of my knee"},
        {"role": "assistant", "content": "Does the pain spread anywhere else?"},
        {"role": "user", "content": "No, it stays in the front of my knee"},
        {"role": "assistant", "content": "Do you have any other symptoms like swelling, stiffness, or weakness?"},
        {"role": "user", "content": "Yes, I have morning stiffness that lasts about 20 minutes"},
        {"role": "assistant", "content": "Is the pain constant or does it come and go?"},
        {"role": "user", "content": "It's worse in the morning and after sitting for a while"},
        {"role": "assistant", "content": "What makes the pain better or worse?"},
        {"role": "user", "content": "Walking helps, but stairs and squatting make it worse"},
        {"role": "assistant", "content": "On a scale of 0 to 10, how severe is your pain?"},
        {"role": "user", "content": "About a 6 out of 10"},
        {"role": "assistant", "content": "How long does the morning stiffness last?"},
        {"role": "user", "content": "About 20-30 minutes"},
        {"role": "assistant", "content": "How is this affecting your daily activities?"},
        {"role": "user", "content": "I have trouble with stairs and getting up from chairs"},
        {"role": "assistant", "content": "Have you tried any treatments before?"},
        {"role": "user", "content": "I've tried paracetamol and some exercises from YouTube"},
        {"role": "assistant", "content": "To make sure we're not missing anything serious, have you experienced any fever, chills, unexplained weight loss, or severe weakness?"},
        {"role": "user", "content": "No, none of those"},
        {"role": "assistant", "content": "Thank you for sharing all the information with me. I now have a complete picture of your situation. A clinical summary with differential diagnosis will be prepared for the clinical team at SWLEOC to direct you to the most appropriate care pathway."}
    ]
    
    # Test questionnaire analysis
    patient_data = agent._extract_patient_data_from_conversation(messages)
    print(f"Extracted patient data: {patient_data}")
    
    questionnaire_analysis = agent._run_questionnaire_analysis(patient_data, "knee_oa")
    print(f"Questionnaire analysis: {questionnaire_analysis}")

def main():
    """Run all tests."""
    print("🚀 Starting MSK Triage System Tests")
    print("=" * 50)
    
    # Test questionnaire engine
    test_questionnaire_engine()
    
    # Test triage agent
    test_triage_agent()
    
    # Test summarization agent
    asyncio.run(test_summarization_agent())
    
    print("\n✅ All tests completed!")
    print("\n📋 Summary of Changes:")
    print("- ✅ Replaced SOCRATES with questionnaire-based assessments")
    print("- ✅ Added questionnaire evaluation engine")
    print("- ✅ Implemented SBAR clinical summary format")
    print("- ✅ Added differential diagnosis with top 3 diagnoses")
    print("- ✅ Updated UI with questionnaire-based flow")
    print("- ✅ Added support for knee OA and knee injury questionnaires")

if __name__ == "__main__":
    main()

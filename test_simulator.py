#!/usr/bin/env python3
"""
Test script for the patient simulator
"""

import asyncio
from patient_simulator import PatientSimulator, PatientData

async def test_with_example_data():
    """Test the simulator with example patient data"""
    
    # Example patient data
    example_patient = PatientData(
        demographics={
            "age": "45",
            "gender": "male", 
            "occupation": "office worker",
            "comorbidities": "none"
        },
        presenting_complaint="Right knee pain for 8 months",
        socrates={
            "site": "Medial aspect of right knee",
            "onset": "Gradual onset over 8 months",
            "character": "Dull, aching pain with occasional sharp episodes",
            "radiation": "No radiation to other areas",
            "associations": "Morning stiffness, occasional swelling",
            "timing": "Worse in morning, improves with activity",
            "exacerbating_relieving": "Worse with stairs, better with heat",
            "severity": "6/10 at worst, 3/10 at best"
        },
        triage_info={
            "injury_mechanism": "No specific injury, gradual onset",
            "red_flags": "None",
            "previous_treatment": "GP consultation, paracetamol",
            "functional_impact": "Difficulty with stairs, affects work"
        },
        expected_triage="MSK Physiotherapy",
        clinical_notes="Typical presentation of medial compartment osteoarthritis"
    )
    
    # Initialize simulator (without OpenAI API key for testing)
    simulator = PatientSimulator()
    simulator.load_patient_data(example_patient)
    
    print("Testing patient simulator with example data...")
    print("Note: This will show the conversation flow but patient responses will be simulated.")
    
    # Run simulation
    await simulator.simulate_conversation()

if __name__ == "__main__":
    asyncio.run(test_with_example_data())

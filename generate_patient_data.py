#!/usr/bin/env python3
"""
Patient Data Generator for MSK Triage Bot Testing

This script generates patient data using GPT and feeds it to the patient simulator.
"""

import asyncio
import json
from openai import OpenAI
from patient_simulator import PatientSimulator, parse_patient_data_from_gpt, PatientData

class PatientDataGenerator:
    """Generates patient data using GPT"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def generate_patient_case(self, triage_pathway: str = "MSK Physiotherapy") -> str:
        """Generate a single patient case using GPT"""
        
        prompt = f"""Generate a realistic patient case that would be triaged to: {triage_pathway}

OUTPUT FORMAT:
**Patient Demographics**
- Age: [age]
- Gender: [gender]
- Occupation: [occupation]
- Comorbidities: [comorbidities]

**Presenting Complaint**
[Main symptom and duration]

**SOCRATES Assessment**
- Site: [exact location]
- Onset: [when and how it started]
- Character: [description of pain/symptom]
- Radiation: [does it spread?]
- Associations: [other symptoms]
- Timing: [pattern and frequency]
- Exacerbating/Relieving: [what makes it better/worse]
- Severity: [0-10 scale]

**Triage Assessment**
- Injury Mechanism: [how it started]
- Red Flags: [serious condition indicators]
- Previous Treatment: [what has been tried]
- Functional Impact: [effect on daily life]

**Triage Recommendation**
{triage_pathway} - [justification]

**Clinical Notes**
[Key clinical reasoning]

Generate a realistic case that would be appropriately triaged to {triage_pathway}."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating patient case: {e}"
    
    def generate_multiple_cases(self, num_cases: int = 5) -> list:
        """Generate multiple patient cases across different triage pathways"""
        
        pathways = [
            "MSK Physiotherapy",
            "GP/Primary Care", 
            "SWLEOC Orthopaedic Surgery",
            "Urgent Care/A&E"
        ]
        
        cases = []
        for i in range(num_cases):
            pathway = pathways[i % len(pathways)]
            print(f"Generating case {i+1}/{num_cases} for {pathway}...")
            case = self.generate_patient_case(pathway)
            cases.append({
                "pathway": pathway,
                "data": case
            })
        
        return cases

async def run_simulation_with_generated_data():
    """Run patient simulator with GPT-generated data"""
    
    print("MSK Triage Bot Patient Simulator with GPT-Generated Data")
    print("=" * 60)
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: OpenAI API key is required!")
        return
    
    # Initialize generator and simulator
    generator = PatientDataGenerator(api_key)
    simulator = PatientSimulator(openai_api_key=api_key)
    
    # Generate patient cases
    print("\nGenerating patient cases...")
    cases = generator.generate_multiple_cases(3)  # Generate 3 cases
    
    # Run simulations for each case
    for i, case in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"SIMULATION {i+1}: {case['pathway']}")
        print(f"{'='*60}")
        
        # Parse patient data
        try:
            patient_data = parse_patient_data_from_gpt(case['data'])
            simulator.load_patient_data(patient_data)
            
            # Run simulation
            await simulator.simulate_conversation()
            
            # Ask if user wants to continue
            if i < len(cases) - 1:
                input("\nPress Enter to continue to next simulation...")
                
        except Exception as e:
            print(f"Error running simulation: {e}")
            continue

async def run_single_simulation():
    """Run a single simulation with user-specified pathway"""
    
    print("MSK Triage Bot Patient Simulator")
    print("=" * 40)
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("Error: OpenAI API key is required!")
        return
    
    # Get triage pathway
    print("\nSelect triage pathway:")
    print("1. MSK Physiotherapy")
    print("2. GP/Primary Care")
    print("3. SWLEOC Orthopaedic Surgery")
    print("4. Urgent Care/A&E")
    
    choice = input("Enter choice (1-4): ").strip()
    pathways = {
        "1": "MSK Physiotherapy",
        "2": "GP/Primary Care", 
        "3": "SWLEOC Orthopaedic Surgery",
        "4": "Urgent Care/A&E"
    }
    
    pathway = pathways.get(choice, "MSK Physiotherapy")
    
    # Initialize generator and simulator
    generator = PatientDataGenerator(api_key)
    simulator = PatientSimulator(openai_api_key=api_key)
    
    # Generate patient case
    print(f"\nGenerating patient case for {pathway}...")
    case_data = generator.generate_patient_case(pathway)
    
    # Parse and run simulation
    try:
        patient_data = parse_patient_data_from_gpt(case_data)
        simulator.load_patient_data(patient_data)
        await simulator.simulate_conversation()
    except Exception as e:
        print(f"Error running simulation: {e}")

if __name__ == "__main__":
    print("Choose simulation mode:")
    print("1. Single simulation")
    print("2. Multiple simulations")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "2":
        asyncio.run(run_simulation_with_generated_data())
    else:
        asyncio.run(run_single_simulation())

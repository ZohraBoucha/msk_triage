#!/usr/bin/env python3
"""
Patient Simulator for MSK Triage Bot Testing

This script simulates patient conversations with the MSK triage bot using LLM-generated
patient data. It shows live conversations in the terminal and can adapt responses
based on the triage bot's questions.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from colorama import init, Fore, Back, Style
import openai
from openai import OpenAI

# Initialize colorama for colored terminal output
init(autoreset=True)

@dataclass
class PatientData:
    """Patient information extracted from GPT-generated data"""
    demographics: Dict[str, str]
    presenting_complaint: str
    socrates: Dict[str, str]
    triage_info: Dict[str, str]
    expected_triage: str
    clinical_notes: str

class PatientSimulator:
    """Simulates a patient conversation with the MSK triage bot"""
    
    def __init__(self, triage_bot_url: str = "http://localhost:8000", 
                 openai_api_key: Optional[str] = None):
        self.triage_bot_url = triage_bot_url
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.conversation_history = []
        self.patient_data = None
        
    def load_patient_data(self, patient_data: PatientData):
        """Load patient data for simulation"""
        self.patient_data = patient_data
        self.conversation_history = []
        
    def create_patient_prompt(self, bot_question: str) -> str:
        """Create a prompt for the patient LLM to respond to bot questions"""
        if not self.patient_data:
            return "I don't have patient data loaded."
            
        prompt = f"""You are a patient with the following medical information:

PATIENT DEMOGRAPHICS:
- Age: {self.patient_data.demographics.get('age', 'Not specified')}
- Gender: {self.patient_data.demographics.get('gender', 'Not specified')}
- Occupation: {self.patient_data.demographics.get('occupation', 'Not specified')}
- Comorbidities: {self.patient_data.demographics.get('comorbidities', 'None')}

PRESENTING COMPLAINT:
{self.patient_data.presenting_complaint}

SOCRATES INFORMATION:
- Site: {self.patient_data.socrates.get('site', 'Not specified')}
- Onset: {self.patient_data.socrates.get('onset', 'Not specified')}
- Character: {self.patient_data.socrates.get('character', 'Not specified')}
- Radiation: {self.patient_data.socrates.get('radiation', 'Not specified')}
- Associations: {self.patient_data.socrates.get('associations', 'Not specified')}
- Timing: {self.patient_data.socrates.get('timing', 'Not specified')}
- Exacerbating/Relieving: {self.patient_data.socrates.get('exacerbating_relieving', 'Not specified')}
- Severity: {self.patient_data.socrates.get('severity', 'Not specified')}

TRIAGE INFORMATION:
- Injury Mechanism: {self.patient_data.triage_info.get('injury_mechanism', 'Not specified')}
- Red Flags: {self.patient_data.triage_info.get('red_flags', 'Not specified')}
- Previous Treatment: {self.patient_data.triage_info.get('previous_treatment', 'Not specified')}
- Functional Impact: {self.patient_data.triage_info.get('functional_impact', 'Not specified')}

CONVERSATION HISTORY:
{self._format_conversation_history()}

CURRENT QUESTION FROM TRIAGE BOT:
"{bot_question}"

INSTRUCTIONS:
- Respond as this patient would naturally respond to the question
- Use the information above to answer accurately
- Be conversational and natural, not clinical
- If the question asks about something not in your information, say you don't know or make a reasonable assumption
- Keep responses concise (1-3 sentences)
- Don't volunteer information not asked for

PATIENT RESPONSE:"""

        return prompt
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for the prompt"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted = []
        for msg in self.conversation_history:
            role = "TRIAGE BOT" if msg['role'] == 'assistant' else "PATIENT"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    async def get_patient_response(self, bot_question: str) -> str:
        """Get patient response using LLM"""
        if not self.client:
            return "I need an OpenAI API key to simulate patient responses."
        
        prompt = self.create_patient_prompt(bot_question)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating patient response: {e}"
    
    async def get_bot_response(self, user_message: str) -> str:
        """Get response from the triage bot"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.triage_bot_url}/ask",
                    json={
                        "messages": self.conversation_history + [{"role": "user", "content": user_message}],
                        "model": "llama3.1:8b"
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "No response from bot")
        except Exception as e:
            return f"Error connecting to triage bot: {e}"
    
    def print_message(self, role: str, content: str, color: str = Fore.WHITE):
        """Print message with color coding"""
        timestamp = time.strftime("%H:%M:%S")
        role_color = Fore.CYAN if role == "BOT" else Fore.GREEN
        print(f"{Fore.YELLOW}[{timestamp}] {role_color}{role}:{Style.RESET_ALL} {content}")
        print()  # Add spacing
    
    async def simulate_conversation(self):
        """Simulate a complete conversation between patient and triage bot"""
        if not self.patient_data:
            print(f"{Fore.RED}Error: No patient data loaded!")
            return
        
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}STARTING PATIENT SIMULATION")
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.BLUE}Patient: {self.patient_data.demographics.get('age', 'Unknown')} year old {self.patient_data.demographics.get('gender', 'person')}")
        print(f"{Fore.BLUE}Occupation: {self.patient_data.demographics.get('occupation', 'Unknown')}")
        print(f"{Fore.BLUE}Expected Triage: {self.patient_data.expected_triage}")
        print(f"{Fore.MAGENTA}{'='*60}")
        print()
        
        # Start with initial patient message
        initial_message = f"I have {self.patient_data.presenting_complaint.lower()}"
        self.conversation_history.append({"role": "user", "content": initial_message})
        self.print_message("PATIENT", initial_message)
        
        # Get bot's first response
        bot_response = await self.get_bot_response(initial_message)
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        self.print_message("BOT", bot_response)
        
        # Continue conversation until completion
        max_exchanges = 20  # Prevent infinite loops
        exchange_count = 0
        
        while exchange_count < max_exchanges:
            # Check if conversation is complete
            if "summary will be prepared" in bot_response.lower():
                print(f"{Fore.GREEN}Conversation completed! Bot is preparing summary...")
                break
            
            # Get patient response to bot's question
            patient_response = await self.get_patient_response(bot_response)
            self.conversation_history.append({"role": "user", "content": patient_response})
            self.print_message("PATIENT", patient_response)
            
            # Get bot's next response
            bot_response = await self.get_bot_response(patient_response)
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            self.print_message("BOT", bot_response)
            
            exchange_count += 1
            
            # Small delay for readability
            await asyncio.sleep(1)
        
        if exchange_count >= max_exchanges:
            print(f"{Fore.YELLOW}Conversation stopped after {max_exchanges} exchanges")
        
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}CONVERSATION COMPLETED")
        print(f"{Fore.MAGENTA}{'='*60}")

def parse_patient_data_from_gpt(gpt_output: str) -> PatientData:
    """Parse patient data from GPT-generated output"""
    # This is a simplified parser - you might want to make it more robust
    lines = gpt_output.split('\n')
    
    demographics = {}
    socrates = {}
    triage_info = {}
    presenting_complaint = ""
    expected_triage = ""
    clinical_notes = ""
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "Patient Demographics" in line:
            current_section = "demographics"
        elif "Presenting Complaint" in line:
            current_section = "presenting"
        elif "SOCRATES Assessment" in line:
            current_section = "socrates"
        elif "Triage Assessment" in line:
            current_section = "triage"
        elif "Triage Recommendation" in line:
            current_section = "recommendation"
        elif "Clinical Notes" in line:
            current_section = "notes"
        elif line.startswith("- "):
            # Parse key-value pairs
            if ":" in line:
                key, value = line[2:].split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                
                if current_section == "demographics":
                    demographics[key] = value
                elif current_section == "socrates":
                    socrates[key] = value
                elif current_section == "triage":
                    triage_info[key] = value
        elif current_section == "presenting" and line:
            presenting_complaint = line
        elif current_section == "recommendation" and line:
            expected_triage = line
        elif current_section == "notes" and line:
            clinical_notes = line
    
    return PatientData(
        demographics=demographics,
        presenting_complaint=presenting_complaint,
        socrates=socrates,
        triage_info=triage_info,
        expected_triage=expected_triage,
        clinical_notes=clinical_notes
    )

async def main():
    """Main function to run the patient simulator"""
    print(f"{Fore.CYAN}MSK Triage Bot Patient Simulator")
    print(f"{Fore.CYAN}{'='*40}")
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    if not api_key:
        print(f"{Fore.YELLOW}No API key provided. Patient responses will be simulated.")
        api_key = None
    
    # Initialize simulator
    simulator = PatientSimulator(openai_api_key=api_key)
    
    # Example patient data (you can replace this with GPT-generated data)
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
    
    # Load patient data
    simulator.load_patient_data(example_patient)
    
    # Start simulation
    await simulator.simulate_conversation()

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Patient Simulator for MSK Triage Bot Testing using Ollama

This script simulates patient conversations with the MSK triage bot using
pre-defined patient cases and Ollama LLM for generating patient responses.
"""

import asyncio
import httpx
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from colorama import init, Fore, Back, Style
import random

# Add the app directory to the path so we can import the agents directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from triage_agent import TriageAgent
from summarization_agent import SummarizationAgent

# Initialize colorama for colored terminal output
init(autoreset=True)

@dataclass
class PatientData:
    """Patient information from the cases file"""
    case_id: str
    title: str
    demographics: Dict[str, str]
    presenting_complaint: str
    socrates: Dict[str, str]
    triage_info: Dict[str, str]
    expected_triage: str
    clinical_notes: str
    sample_conversation: List[str]
    expanded_clinic_letter: str = ""

class PatientSimulator:
    """Simulates a patient conversation with the MSK triage bot using Ollama"""
    
    def __init__(self, triage_bot_url: str = "http://localhost:8000", 
                 ollama_url: str = "http://localhost:11434"):
        self.triage_bot_url = triage_bot_url
        self.ollama_url = ollama_url
        self.conversation_history = []
        self.patient_data = None
        self.conversation_index = 0
        self.conversation_log = []  # Store all conversation messages for saving
        self.generated_summary = ""  # Store the generated SBAR summary
        
        # Initialize the agents directly
        self.triage_agent = TriageAgent(model="llama3.1:8b")
        self.summarization_agent = SummarizationAgent(model="llama3.1:8b")
        
    def load_patient_data(self, patient_data: PatientData):
        """Load patient data for simulation"""
        self.patient_data = patient_data
        self.conversation_history = []
        self.conversation_log = []
        self.conversation_index = 0
        
        # Reset the triage agent state for each new simulation
        self.triage_agent = TriageAgent(model="llama3.1:8b")
        self.summarization_agent = SummarizationAgent(model="llama3.1:8b")
        
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

CLINICAL INFORMATION (SOCRATES):
- Site: {self.patient_data.socrates.get('site', 'Not specified')}
- Onset: {self.patient_data.socrates.get('onset', 'Not specified')}
- Character: {self.patient_data.socrates.get('character', 'Not specified')}
- Radiation: {self.patient_data.socrates.get('radiation', 'Not specified')}
- Associations: {self.patient_data.socrates.get('associations', 'Not specified')}
- Timing: {self.patient_data.socrates.get('timing', 'Not specified')}
- Exacerbating/Relieving: {self.patient_data.socrates.get('exacerbating', 'Not specified')} / {self.patient_data.socrates.get('relieving', 'Not specified')}
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
- Use the information above to answer accurately and consistently
- Be conversational and natural, not clinical
- If the question asks about something not in your information, say you don't know or make a reasonable assumption based on the clinical information provided
- Keep responses concise (1-3 sentences)
- Don't volunteer information not asked for
- Don't repeat the bot's question back to them
- Don't repeat previous answers unless specifically asked
- ALWAYS respond to the bot's question - never leave it unanswered
- Use the sample conversation responses as a guide for style and content
- For age questions, use the exact age from demographics
- For body part questions, use the site information from SOCRATES
- For duration questions, use the onset information to determine if it's acute, subacute, or chronic
- For mechanism questions, use the injury mechanism information
- For pain questions, use the character, severity, and timing information
- For functional impact questions, use the functional impact information
- If asked for demographics, provide: "I'm [age] years old, [gender], and my date of birth is [make up a realistic DOB]"

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
        """Get patient response using Ollama"""
        prompt = self.create_patient_prompt(bot_question)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 100
                        }
                    }
                )
                response.raise_for_status()
                ollama_response = response.json()
                return ollama_response.get("response", "No response from patient LLM.").strip()
        except Exception as e:
            return f"Error generating patient response: {e}"
    
    async def get_bot_response(self, user_message: str) -> str:
        """Get response from the triage bot"""
        try:
            # Add the user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Use the triage agent directly
            response = await self.triage_agent.get_next_response(self.conversation_history)
            
            # Add the bot response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"Error getting bot response: {e}"
    
    async def generate_summary(self):
        """Generate clinical summary using the summarization agent directly"""
        try:
            print(f"{Fore.CYAN}Generating clinical summary...")
            
            # Use the summarization agent directly
            summary = await self.summarization_agent.summarize_and_triage(self.conversation_history)
            
            # Store the summary for saving to file
            self.generated_summary = summary
            
            print(f"{Fore.MAGENTA}{'='*60}")
            print(f"{Fore.MAGENTA}SBAR CLINICAL SUMMARY & DIFFERENTIAL DIAGNOSIS")
            print(f"{Fore.MAGENTA}{'='*60}")
            print(f"{Fore.WHITE}{summary}")
            print(f"{Fore.MAGENTA}{'='*60}")
            
        except Exception as e:
            print(f"{Fore.RED}Error generating summary: {e}")
    
    def print_message(self, role: str, content: str, color: str = Fore.WHITE):
        """Print message with color coding and log it"""
        timestamp = time.strftime("%H:%M:%S")
        role_color = Fore.CYAN if role == "BOT" else Fore.GREEN
        print(f"{Fore.YELLOW}[{timestamp}] {role_color}{role}:{Style.RESET_ALL} {content}")
        print()  # Add spacing
        
        # Log the message for saving (clean content without developer notes)
        clean_content = content
        # Remove any developer notes in parentheses
        import re
        clean_content = re.sub(r'\(Note:.*?\)', '', clean_content)
        clean_content = re.sub(r'\(This response is based on.*?\)', '', clean_content)
        clean_content = clean_content.strip()
        
        self.conversation_log.append({
            "timestamp": timestamp,
            "role": role,
            "content": clean_content
        })
    
    def save_conversation_to_file(self, output_dir: str = "conversation_logs"):
        """Save the conversation to a text file"""
        if not self.patient_data or not self.conversation_log:
            print(f"{Fore.RED}No conversation data to save!")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp and case info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = self.patient_data.case_id.replace("case_", "")
        filename = f"{timestamp}_{case_id}_{self.patient_data.title.replace(' ', '_')}.txt"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write header information
                f.write("MSK TRIAGE BOT CONVERSATION LOG (QUESTIONNAIRE-BASED)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Case ID: {self.patient_data.case_id}\n")
                f.write(f"Title: {self.patient_data.title}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Assessment Type: Questionnaire-Based MSK Triage\n")
                f.write(f"Output Format: SBAR Clinical Summary + Top 3 Differential Diagnoses\n\n")
                
                # Write patient demographics
                f.write("PATIENT DEMOGRAPHICS:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.patient_data.demographics.items():
                    f.write(f"{key.title()}: {value}\n")
                f.write(f"Expected Triage: {self.patient_data.expected_triage}\n\n")
                
                # Write presenting complaint
                f.write("PRESENTING COMPLAINT:\n")
                f.write("-" * 20 + "\n")
                f.write(f"{self.patient_data.presenting_complaint}\n\n")
                
                # Write conversation
                f.write("CONVERSATION:\n")
                f.write("-" * 20 + "\n")
                for msg in self.conversation_log:
                    f.write(f"[{msg['timestamp']}] {msg['role']}: {msg['content']}\n\n")
                
                # Write clinical notes if available
                if self.patient_data.clinical_notes:
                    f.write("CLINICAL NOTES:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"{self.patient_data.clinical_notes}\n\n")
                
                # Write expanded clinic letter if available
                if hasattr(self.patient_data, 'expanded_clinic_letter') and self.patient_data.expanded_clinic_letter:
                    f.write("EXPANDED CLINIC LETTER:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"{self.patient_data.expanded_clinic_letter}\n\n")
                
                # Write SBAR summary and triage classification (excluding differential diagnosis)
                if self.generated_summary:
                    f.write("SBAR CLINICAL SUMMARY & TRIAGE CLASSIFICATION:\n")
                    f.write("=" * 60 + "\n")
                    
                    # Split the summary into sections
                    sections = self.generated_summary.split("---")
                    if len(sections) >= 1:
                        # Write the SBAR section (first part before the first ---)
                        f.write(sections[0].strip())
                        f.write("\n\n")
                    
                    # Find and write the triage classification section
                    triage_section = None
                    for section in sections[1:]:  # Skip the first section (SBAR)
                        if "TRIAGE CLASSIFICATION:" in section:
                            triage_section = section
                            break
                    
                    if triage_section:
                        f.write("---\n\n")
                        f.write(triage_section.strip())
                        f.write("\n\n")
            
            print(f"{Fore.GREEN}Conversation saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"{Fore.RED}Error saving conversation: {e}")
            return None
    
    async def simulate_conversation(self):
        """Simulate a complete conversation between patient and triage bot"""
        if not self.patient_data:
            print(f"{Fore.RED}Error: No patient data loaded!")
            return
        
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}STARTING QUESTIONNAIRE-BASED PATIENT SIMULATION")
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.BLUE}Case: {self.patient_data.title}")
        print(f"{Fore.BLUE}Patient: {self.patient_data.demographics.get('age', 'Unknown')} year old {self.patient_data.demographics.get('gender', 'person')}")
        print(f"{Fore.BLUE}Occupation: {self.patient_data.demographics.get('occupation', 'Unknown')}")
        print(f"{Fore.BLUE}Expected Triage: {self.patient_data.expected_triage}")
        print(f"{Fore.CYAN}Assessment: Questionnaire-Based MSK Triage")
        print(f"{Fore.CYAN}Output: SBAR Summary + Top 3 Differential Diagnoses")
        print(f"{Fore.MAGENTA}{'='*60}")
        print()
        
        # Start with bot's greeting
        bot_response = await self.get_bot_response("")
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        self.print_message("BOT", bot_response)
        
        # Generate realistic name and DOB
        import random
        first_names = ["John", "Sarah", "Michael", "Emma", "David", "Lisa", "James", "Anna", "Robert", "Maria", "William", "Jennifer", "Richard", "Linda", "Charles", "Elizabeth", "Joseph", "Patricia", "Thomas", "Susan", "Christopher", "Jessica", "Daniel", "Sarah", "Matthew", "Ashley", "Anthony", "Emily", "Mark", "Michelle"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        age = int(self.patient_data.demographics.get('age', '30'))
        current_year = 2024
        birth_year = current_year - age
        dob = f"{random.randint(1, 28)}/{random.randint(1, 12)}/{birth_year}"
        
        # Patient responds to greeting with demographics and initial complaint
        initial_message = f"I'm {first_name} {last_name}, {age} years old, {self.patient_data.demographics.get('gender', 'person')}, and my date of birth is {dob}. I have {self.patient_data.presenting_complaint.lower()}"
        self.conversation_history.append({"role": "user", "content": initial_message})
        self.print_message("PATIENT", initial_message)
        
        # Get bot's response to patient's complaint
        bot_response = await self.get_bot_response(initial_message)
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        self.print_message("BOT", bot_response)
        
        # Continue conversation until completion
        max_exchanges = 20  # Prevent infinite loops
        exchange_count = 0
        
        while exchange_count < max_exchanges:
            # Check if conversation is complete
            if "clinical summary with differential diagnosis will be prepared" in bot_response.lower() or "summary will be prepared" in bot_response.lower():
                print(f"{Fore.GREEN}Conversation completed! Bot is preparing SBAR clinical summary and differential diagnosis...")
                
                # Generate the clinical summary
                await self.generate_summary()
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
        
        # Save conversation to file
        saved_file = self.save_conversation_to_file()
        if saved_file:
            print(f"{Fore.CYAN}Conversation log saved successfully!")

def load_patient_cases(file_path: str) -> List[PatientData]:
    """Load patient cases from JSON file"""
    with open(file_path, 'r') as f:
        cases_data = json.load(f)
    
    cases = []
    for case_data in cases_data:
        case = PatientData(
            case_id=case_data['case_id'],
            title=case_data['title'],
            demographics=case_data['demographics'],
            presenting_complaint=case_data['presenting_complaint'],
            socrates=case_data['socrates'],
            triage_info=case_data['triage_info'],
            expected_triage=case_data['expected_triage'],
            clinical_notes=case_data['clinical_notes'],
            sample_conversation=case_data['sample_conversation'],
            expanded_clinic_letter=case_data.get('expanded_clinic_letter', '')
        )
        cases.append(case)
    
    return cases

async def run_single_simulation():
    """Run a single simulation with a randomly selected case"""
    
    print("MSK Triage Bot Patient Simulator (Ollama) - Questionnaire-Based")
    print("=" * 60)
    
    # Load patient cases
    try:
        cases = load_patient_cases('patient_cases.json')
        print(f"Loaded {len(cases)} patient cases")
    except Exception as e:
        print(f"Error loading patient cases: {e}")
        return
    
    # Select a random case
    selected_case = random.choice(cases)
    
    # Initialize simulator
    simulator = PatientSimulator()
    simulator.load_patient_data(selected_case)
    
    # Start simulation
    await simulator.simulate_conversation()

async def run_all_simulations():
    """Run simulations for all patient cases"""
    
    print("MSK Triage Bot Patient Simulator (Ollama) - All Cases (Questionnaire-Based)")
    print("=" * 70)
    
    # Load patient cases
    try:
        cases = load_patient_cases('patient_cases.json')
        print(f"Loaded {len(cases)} patient cases")
    except Exception as e:
        print(f"Error loading patient cases: {e}")
        return
    
    # Initialize simulator
    simulator = PatientSimulator()
    
    # Run simulation for each case
    for i, case in enumerate(cases):
        print(f"\n{'='*60}")
        print(f"SIMULATION {i+1}/{len(cases)}: {case.title}")
        print(f"{'='*60}")
        
        simulator.load_patient_data(case)
        await simulator.simulate_conversation()
        
        # Ask if user wants to continue
        if i < len(cases) - 1:
            input("\nPress Enter to continue to next simulation...")

if __name__ == "__main__":
    print("Choose simulation mode:")
    print("1. Single random case")
    print("2. All cases")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "2":
        asyncio.run(run_all_simulations())
    else:
        asyncio.run(run_single_simulation())

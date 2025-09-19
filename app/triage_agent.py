import httpx
from enum import Enum
from typing import List, Dict, Optional, Any
from questionnaire_specs import get_questionnaire_form, get_available_forms
from questionnaire_engine import run_questionnaire_engine, map_mechanism_from_text

# --- State Machine Definition (Questionnaire-Based) ---
class TriageState(str, Enum):
    # Initial Assessment
    GREETING = "GREETING"
    SELECT_BODY_PART = "SELECT_BODY_PART"
    SELECT_QUESTIONNAIRE = "SELECT_QUESTIONNAIRE"
    
    # Questionnaire Questions
    GATHER_AGE = "GATHER_AGE"
    GATHER_LATERALITY = "GATHER_LATERALITY"
    GATHER_DURATION = "GATHER_DURATION"
    GATHER_MECHANISM = "GATHER_MECHANISM"
    GATHER_SYMPTOMS = "GATHER_SYMPTOMS"
    GATHER_PAIN_CHARACTER = "GATHER_PAIN_CHARACTER"
    GATHER_RADIATION = "GATHER_RADIATION"
    GATHER_ASSOCIATED_SYMPTOMS = "GATHER_ASSOCIATED_SYMPTOMS"
    GATHER_TIMING = "GATHER_TIMING"
    GATHER_EXACERBATING_RELIEVING = "GATHER_EXACERBATING_RELIEVING"
    GATHER_SEVERITY = "GATHER_SEVERITY"
    GATHER_STIFFNESS = "GATHER_STIFFNESS"
    GATHER_FUNCTIONAL_IMPACT = "GATHER_FUNCTIONAL_IMPACT"
    GATHER_PREVIOUS_TREATMENT = "GATHER_PREVIOUS_TREATMENT"
    GATHER_RED_FLAGS = "GATHER_RED_FLAGS"
    
    # Specialized Questions (Knee Injury)
    GATHER_KNEE_SCORE = "GATHER_KNEE_SCORE"
    GATHER_EXAM_FINDINGS = "GATHER_EXAM_FINDINGS"
    GATHER_IMAGING = "GATHER_IMAGING"
    
    # Completion
    COMPLETE = "COMPLETE"

# --- Triage Agent Class ---
class TriageAgent:
    """
    Manages the state and logic of the triage conversation using questionnaire-based assessments.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://localhost:11434/api/generate"
        self.current_questionnaire = None
        self.patient_data = {}
        self.question_index = 0
        
        self.system_prompt_template = """You are Leo, a professional AI assistant for the Southwest London Elective Orthopaedic Centre (SWLEOC).
Your job is to carry out an initial musculoskeletal assessment using structured questionnaires.

CRITICAL RULES:
• Remain empathetic and professional at all times.
• You are gathering information only, not providing diagnoses or medical advice.
• Follow the EXACT instructions provided in the Question section below.
• If the instruction says "Say exactly:", you must use those exact words.
• Ask ONLY the next question listed below. Do not ask multiple questions at once.
• You may rephrase questions slightly to make them flow naturally, UNLESS the instruction says "Say exactly:".
• NEVER add parenthetical notes, internal thoughts, or meta-commentary to your responses.
• NEVER use phrases like "(Note:...)" or "(Also, I'll...)" or any parenthetical explanations.
• Stay in character as a medical professional at all times.
• Keep responses concise and focused on the current question.
• Do not reference previous questions or explain your process.
• Respond as if you are speaking directly to the patient, not documenting your thoughts.
• Don't repeat information the patient has already provided.
• Don't ask the same question twice.
• If the patient has already answered a question, acknowledge it and move to the next question.

**Question:** {current_task_prompt}
"""

    def _get_prompt_for_state(self, state: TriageState) -> str:
        """Returns the GOAL for the AI for a given state."""
        prompts = {
            # --- Initial Assessment ---
            TriageState.GREETING: "Say exactly: 'Hello, I'm Leo, your musculoskeletal triage assistant. I'm here to help assess your condition and direct you to the most appropriate care. To get started, could you please tell me your full name, age, gender, and date of birth?'",
            TriageState.SELECT_BODY_PART: "Say exactly: 'Thank you. Now, what brings you here today? Which part of your body is affected (e.g., knee, shoulder, back, hip, etc.)?'",
            TriageState.SELECT_QUESTIONNAIRE: "Based on the body part mentioned, ask if this is related to an injury/accident or if it's a more gradual onset problem.",
            
            # --- Basic Demographics ---
            TriageState.GATHER_AGE: "Ask the patient for their age if not already provided.",
            TriageState.GATHER_LATERALITY: "Ask which side is affected (left or right).",
            
            # --- Onset and Mechanism ---
            TriageState.GATHER_DURATION: "Ask how long they have had this problem (acute: <2 weeks, subacute: 2-12 weeks, chronic: >12 weeks).",
            TriageState.GATHER_MECHANISM: "Ask how the problem started - was it from an injury, overuse, or did it come on gradually?",
            
            # --- Symptoms ---
            TriageState.GATHER_SYMPTOMS: "Ask them to describe their main symptoms in their own words.",
            TriageState.GATHER_PAIN_CHARACTER: "Ask them to describe what the pain feels like (sharp, dull, aching, burning, etc.).",
            TriageState.GATHER_RADIATION: "Ask if the pain or feeling spreads to any other part of their body.",
            TriageState.GATHER_ASSOCIATED_SYMPTOMS: "Ask if they have any other symptoms like swelling, stiffness, numbness, or weakness.",
            TriageState.GATHER_TIMING: "Ask if the symptoms are constant or come and go, and if there's any pattern to them.",
            TriageState.GATHER_EXACERBATING_RELIEVING: "Ask what makes the symptoms better or worse.",
            TriageState.GATHER_SEVERITY: "Ask them to rate their pain on a scale of 0 to 10, where 0 is no pain and 10 is the worst imaginable.",
            
            # --- Stiffness (for OA) ---
            TriageState.GATHER_STIFFNESS: "Ask about morning stiffness - how long does it take to loosen up in the morning?",
            
            # --- Functional Impact ---
            TriageState.GATHER_FUNCTIONAL_IMPACT: "Ask how this problem is affecting their daily activities, work, or hobbies.",
            TriageState.GATHER_PREVIOUS_TREATMENT: "Ask what treatments they have tried before (medications, physiotherapy, etc.).",
            TriageState.GATHER_RED_FLAGS: "Ask an important safety question: 'To make sure we're not missing anything serious, have you experienced any fever, chills, unexplained weight loss, or severe weakness?'",
            
            # --- Specialized Questions (Knee Injury) ---
            TriageState.GATHER_KNEE_SCORE: "Ask them to rate their knee function on various activities (walking, stairs, squatting, etc.).",
            TriageState.GATHER_EXAM_FINDINGS: "Ask about any physical examination findings they may know about (swelling, instability, etc.).",
            TriageState.GATHER_IMAGING: "Ask if they have had any imaging done (X-rays, MRI) and what the results were.",
            
            # --- Completion ---
            TriageState.COMPLETE: "Thank the user for all the information. State that you now have a complete picture of their situation and that a clinical summary with differential diagnosis will be prepared for the clinical team to direct them to the most appropriate care pathway."
        }
        return prompts.get(state, "The conversation is complete.")

    def _determine_current_state(self, messages: List[Dict]) -> TriageState:
        """Determines the current state based on the conversation history and questionnaire type."""
        num_user_answers = len([msg for msg in messages if msg['role'] == 'user']) - 1
        
        # Determine questionnaire type based on conversation
        if not self.current_questionnaire:
            # Look for body part and injury type in conversation
            last_user_message = messages[-1]['content'].lower() if messages else ""
            
            if 'knee' in last_user_message:
                if any(word in last_user_message for word in ['injury', 'hurt', 'injured', 'accident', 'fall', 'twist']):
                    self.current_questionnaire = 'knee_injury'
                else:
                    self.current_questionnaire = 'knee_oa'
            else:
                # Default to knee OA for now
                self.current_questionnaire = 'knee_oa'
        
        # Define state sequences based on questionnaire type
        if self.current_questionnaire == 'knee_injury':
            state_sequence = [
                TriageState.GREETING,
                TriageState.SELECT_BODY_PART,
                TriageState.SELECT_QUESTIONNAIRE,
                TriageState.GATHER_AGE,
                TriageState.GATHER_LATERALITY,
                TriageState.GATHER_DURATION,
                TriageState.GATHER_MECHANISM,
                TriageState.GATHER_SYMPTOMS,
                TriageState.GATHER_PAIN_CHARACTER,
                TriageState.GATHER_RADIATION,
                TriageState.GATHER_ASSOCIATED_SYMPTOMS,
                TriageState.GATHER_TIMING,
                TriageState.GATHER_EXACERBATING_RELIEVING,
                TriageState.GATHER_SEVERITY,
                TriageState.GATHER_KNEE_SCORE,
                TriageState.GATHER_FUNCTIONAL_IMPACT,
                TriageState.GATHER_PREVIOUS_TREATMENT,
                TriageState.GATHER_RED_FLAGS,
                TriageState.COMPLETE
            ]
        else:  # knee_oa
            state_sequence = [
                TriageState.GREETING,
                TriageState.SELECT_BODY_PART,
                TriageState.SELECT_QUESTIONNAIRE,
                TriageState.GATHER_AGE,
                TriageState.GATHER_LATERALITY,
                TriageState.GATHER_DURATION,
                TriageState.GATHER_MECHANISM,
                TriageState.GATHER_SYMPTOMS,
                TriageState.GATHER_PAIN_CHARACTER,
                TriageState.GATHER_RADIATION,
                TriageState.GATHER_ASSOCIATED_SYMPTOMS,
                TriageState.GATHER_TIMING,
                TriageState.GATHER_EXACERBATING_RELIEVING,
                TriageState.GATHER_SEVERITY,
                TriageState.GATHER_STIFFNESS,
                TriageState.GATHER_FUNCTIONAL_IMPACT,
                TriageState.GATHER_PREVIOUS_TREATMENT,
                TriageState.GATHER_RED_FLAGS,
                TriageState.COMPLETE
            ]
        
        # Adjust index for the new flow (GREETING and SELECT_BODY_PART are now at the beginning)
        if num_user_answers < len(state_sequence):
            return state_sequence[num_user_answers]
        return TriageState.COMPLETE

    def _extract_patient_data(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract structured patient data from conversation messages."""
        data = {
            "patient": {},
            "laterality": None,
            "duration_class": None,
            "mechanism": None,
            "phenotype": [],
            "oa_index": {},
            "knee_score": {},
            "exam": {},
            "imaging": {},
            "red_flags": {
                "fever_unwell_hot_joint": False,
                "true_locked_knee": False,
                "inability_slr_after_eccentric_load": False
            }
        }
        
        # Extract data from user messages
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                
                # Extract age
                if any(word in content for word in ['age', 'years old', 'i am', 'i\'m']):
                    import re
                    age_match = re.search(r'(\d+)', content)
                    if age_match:
                        data["patient"]["age_years"] = int(age_match.group(1))
                
                # Extract gender
                if any(word in content for word in ['female', 'woman', 'girl', 'she', 'her']):
                    data["patient"]["gender"] = "female"
                elif any(word in content for word in ['male', 'man', 'boy', 'he', 'him']):
                    data["patient"]["gender"] = "male"
                
                # Extract laterality
                if 'left' in content:
                    data["laterality"] = "left"
                elif 'right' in content:
                    data["laterality"] = "right"
                
                # Extract duration
                if any(word in content for word in ['acute', 'recent', 'just', 'today', 'yesterday']):
                    data["duration_class"] = "acute"
                elif any(word in content for word in ['chronic', 'months', 'years', 'long time']):
                    data["duration_class"] = "chronic"
                elif any(word in content for word in ['weeks', 'subacute']):
                    data["duration_class"] = "subacute"
                
                # Extract mechanism
                if any(word in content for word in ['injury', 'hurt', 'injured', 'accident', 'fall', 'twist']):
                    data["mechanism"] = "twisting"
                elif any(word in content for word in ['overuse', 'gradual', 'insidious']):
                    data["mechanism"] = "overuse"
                elif any(word in content for word in ['blow', 'contact', 'collision']):
                    data["mechanism"] = "direct_blow"
                
                # Extract symptoms/phenotype
                if 'instability' in content or 'giving way' in content:
                    data["phenotype"].append("instability")
                if 'locking' in content or 'catching' in content:
                    data["phenotype"].append("locking_catching")
                if 'anterior' in content or 'front' in content:
                    data["phenotype"].append("anterior_pain")
                
                # Extract red flags
                if any(word in content for word in ['fever', 'hot', 'unwell', 'sick']):
                    data["red_flags"]["fever_unwell_hot_joint"] = True
                if 'locked' in content and 'knee' in content:
                    data["red_flags"]["true_locked_knee"] = True
                if 'straight leg raise' in content or 'slr' in content:
                    data["red_flags"]["inability_slr_after_eccentric_load"] = True
        
        return data

    async def get_next_response(self, messages: List[Dict]) -> str:
        """
        Takes the conversation history, determines the next step,
        and gets a response from the LLM.
        """
        current_state = self._determine_current_state(messages)
        
        # If we're in COMPLETE state, return a fixed completion message
        if current_state == TriageState.COMPLETE:
            return "Thank you for sharing all the information with me. I now have a complete picture of your situation. A clinical summary with differential diagnosis will be prepared for the clinical team at SWLEOC to direct you to the most appropriate care pathway."
        
        task_prompt = self._get_prompt_for_state(current_state)
        system_prompt = self.system_prompt_template.format(current_task_prompt=task_prompt)
        
        full_prompt = f"System: {system_prompt}\n\n"
        for msg in messages:
            full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        full_prompt += "Assistant:"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                return ollama_response.get("response", "No response content.").strip()
        except httpx.RequestError as e:
            print(f"Error connecting to Ollama: {e}")
            return "I'm sorry, but I'm having trouble connecting to my services right now."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "I'm sorry, an unexpected error occurred."
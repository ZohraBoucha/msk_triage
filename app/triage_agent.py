import httpx
from enum import Enum
from typing import List, Dict

# --- State Machine Definition (Refactored for SOCRATES) ---
class TriageState(str, Enum):
    GREETING = "GREETING"
    GATHER_SITE = "GATHER_SITE"  # S: Site
    GATHER_ONSET = "GATHER_ONSET"  # O: Onset
    GATHER_CHARACTER = "GATHER_CHARACTER"  # C: Character
    GATHER_RADIATION = "GATHER_RADIATION"  # R: Radiation
    GATHER_ASSOCIATIONS = "GATHER_ASSOCIATIONS"  # A: Associations
    GATHER_TIMING = "GATHER_TIMING"  # T: Time course
    GATHER_EXACERBATING_RELIEVING = "GATHER_EXACERBATING_RELIEVING"  # E: Exacerbating/Relieving
    GATHER_SEVERITY = "GATHER_SEVERITY"  # S: Severity
    COMPLETE = "COMPLETE"

# --- Triage Agent Class ---
class TriageAgent:
    """
    Manages the state and logic of the triage conversation.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://llm_server:11434/api/generate"
        self.system_prompt_template = """You are Leo, a professional AI assistant for the Southwest London Elective Orthopaedic Centre (SWLEOC).
Your job is to carry out an initial musculoskeletal assessment by asking a series of questions.

Follow these rules:
• Remain empathetic and professional at all times.
• You may thank the patient or express understanding when they describe a symptom or difficulty, but you do not need to acknowledge every single answer.
• You are gathering information only, not providing diagnoses or medical advice.
• Ask the next question listed below. You may rephrase it slightly to make it flow naturally in the conversation, but do not change its meaning or add extra information.

**Question:** {current_task_prompt}
"""

    def _get_prompt_for_state(self, state: TriageState) -> str:
        """Returns the GOAL for the AI for a given state."""
        prompts = {
            # S: Site
            TriageState.GATHER_SITE: "Acknowledge the user's main symptom, then ask them to specify exactly where in their body the problem is.",
            # O: Onset
            TriageState.GATHER_ONSET: "Acknowledge their last answer. Now, ask them when the problem started and if it came on suddenly or gradually.",
            # C: Character
            TriageState.GATHER_CHARACTER: "Acknowledge their last answer. Now, ask them to describe what the symptom feels like (e.g., sharp, dull, aching, burning).",
            # R: Radiation
            TriageState.GATHER_RADIATION: "Acknowledge their last answer. Now, ask if the feeling moves or radiates to any other part of their body.",
            # A: Associations
            TriageState.GATHER_ASSOCIATIONS: "Acknowledge their last answer. Now, ask if they have noticed any other symptoms that seem to occur at the same time, like swelling, stiffness, or numbness.",
            # T: Time Course / Pattern
            TriageState.GATHER_TIMING: "Acknowledge their last answer. Now, ask if the symptom is constant or if it comes and goes. Ask if there's any pattern to it (e.g., worse in the morning).",
            # E: Exacerbating / Relieving Factors
            TriageState.GATHER_EXACERBATING_RELIEVING: "Acknowledge their last answer. Now, ask if anything they do makes the symptom better or worse.",
            # S: Severity
            TriageState.GATHER_SEVERITY: "Acknowledge their last answer. Now, ask them to rate their pain or discomfort on a scale of 0 to 10, where 0 is no discomfort and 10 is the worst imaginable.",
            # Completion
            TriageState.COMPLETE: "Thank the user for all the information and state that a summary will be prepared for the clinical team."
        }
        return prompts.get(state, "The conversation is complete.")

    def _determine_current_state(self, messages: List[Dict]) -> TriageState:
        """Determines the current state based on the conversation history."""
        # Count messages from the user, ignoring the initial symptom description
        num_user_answers = len([msg for msg in messages if msg['role'] == 'user']) -1

        state_sequence = [
            TriageState.GATHER_SITE,
            TriageState.GATHER_ONSET,
            TriageState.GATHER_CHARACTER,
            TriageState.GATHER_RADIATION,
            TriageState.GATHER_ASSOCIATIONS,
            TriageState.GATHER_TIMING,
            TriageState.GATHER_EXACERBATING_RELIEVING,
            TriageState.GATHER_SEVERITY,
            TriageState.COMPLETE
        ]
        
        if num_user_answers < len(state_sequence):
            return state_sequence[num_user_answers]
        return TriageState.COMPLETE

    async def get_next_response(self, messages: List[Dict]) -> str:
        """
        Takes the conversation history, determines the next step,
        and gets a response from the LLM.
        """
        current_state = self._determine_current_state(messages)
        task_prompt = self._get_prompt_for_state(current_state)
        system_prompt = self.system_prompt_template.format(current_task_prompt=task_prompt)
        
        full_prompt = f"System: {system_prompt}\n\n"
        for msg in messages:
            full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        full_prompt += "Assistant:"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
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
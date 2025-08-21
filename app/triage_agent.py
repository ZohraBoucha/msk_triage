import httpx
from enum import Enum
from typing import List, Dict

# --- State Machine Definition ---
class TriageState(str, Enum):
    GREETING = "GREETING"
    GATHER_WHICH_KNEE = "GATHER_WHICH_KNEE"
    GATHER_MAIN_BOTHER = "GATHER_MAIN_BOTHER"
    GATHER_PAIN_TIMING = "GATHER_PAIN_TIMING"
    GATHER_PAIN_QUALITY = "GATHER_PAIN_QUALITY"
    GATHER_PAIN_SCALE = "GATHER_PAIN_SCALE"
    GATHER_ONSET_DURATION = "GATHER_ONSET_DURATION"
    COMPLETE = "COMPLETE"

# --- Triage Agent Class ---
class TriageAgent:
    """
    Manages the state and logic of the triage conversation.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://llm_server:11434/api/generate"
        self.system_prompt_template = """You are a helpful and professional AI assistant for the Southwest London Elective Orthopaedic Centre (SWLEOC). Your name is Leo.
Your primary goal is to conduct an initial triage of patients by asking a series of targeted questions based on a clinical questionnaire.

Follow these rules strictly:
1.  Be empathetic and professional at all times.
2.  Your task is to gather information, not to provide a diagnosis or medical advice.
3.  Use the conversation history to ask questions naturally. Do not repeat questions or ask for information the user has already provided.
4.  Your current goal is to: **{current_task_prompt}**
"""

    def _get_prompt_for_state(self, state: TriageState) -> str:
        """Returns the GOAL for the AI for a given state."""
        prompts = {
            TriageState.GATHER_WHICH_KNEE: "Acknowledge the user's main symptom, then ask which knee is giving them problems (Right or Left).",
            TriageState.GATHER_MAIN_BOTHER: "Acknowledge their last answer. Now, find out what is bothering them the most. Guide them toward one of these options if they are unsure: Pain, Dislocation, difficulty with Stairs, or difficulty with Sitting.",
            TriageState.GATHER_PAIN_TIMING: "Acknowledge their last answer. Now, ask if their pain is Constant or Intermittent.",
            TriageState.GATHER_PAIN_QUALITY: "Acknowledge their last answer. Now, ask if the pain is Sharp, Dull, or has a Burning quality.",
            TriageState.GATHER_PAIN_SCALE: "Acknowledge their last answer. Now, ask them to rate their pain over the past week on a scale of 0 to 10, where 0 is no pain and 10 is the most pain imaginable.",
            TriageState.GATHER_ONSET_DURATION: "Acknowledge their last answer. Now, ask them how long their knee has been bothering them.",
            TriageState.COMPLETE: "Thank the user for all the information and state that a summary will be prepared for the clinical team."
        }
        return prompts.get(state, "The conversation is complete.")

    def _determine_current_state(self, messages: List[Dict]) -> TriageState:
        """Determines the current state based on the conversation history."""
        num_user_messages = len([msg for msg in messages if msg['role'] == 'user'])
        
        state_sequence = [
            TriageState.GATHER_WHICH_KNEE,
            TriageState.GATHER_MAIN_BOTHER,
            TriageState.GATHER_PAIN_TIMING,
            TriageState.GATHER_PAIN_QUALITY,
            TriageState.GATHER_PAIN_SCALE,
            TriageState.GATHER_ONSET_DURATION,
            TriageState.COMPLETE
        ]
        
        if num_user_messages < len(state_sequence):
            return state_sequence[num_user_messages]
        return TriageState.COMPLETE

    async def get_next_response(self, messages: List[Dict]) -> str:
        """
        Takes the conversation history, determines the next step,
        and gets a response from the LLM.
        """
        current_state = self._determine_current_state(messages)
        task_prompt = self._get_prompt_for_state(current_state)
        system_prompt = self.system_prompt_template.format(current_task_prompt=task_prompt)
        
        # **RE-INTRODUCING CONVERSATION HISTORY**
        # This is the key change that allows the AI to be more natural.
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

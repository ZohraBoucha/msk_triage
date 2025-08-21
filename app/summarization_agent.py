import httpx
from typing import List, Dict

class SummarizationAgent:
    """
    Analyzes a conversation transcript to produce a clinical summary and triage recommendation.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://llm_server:11434/api/generate"
        self.system_prompt_template = """You are an expert Orthopaedic Triage Clinician. Your task is to analyze the following patient-AI conversation and generate a structured clinical summary and a triage recommendation.

Follow these rules:
1.  **Analyze the Full Transcript:** Base your entire analysis on the conversation provided.
2.  **Generate a Structured Summary:** Create a summary using the following Markdown format:
    ---
    **Clinical Summary**
    - **Presenting Complaint:**
    - **History of Presenting Complaint (SOCRATES):**
        - **Site:**
        - **Onset:**
        - **Character:**
        - **Radiation:**
        - **Associated Symptoms:**
        - **Timing:**
        - **Exacerbating/Relieving Factors:**
        - **Severity:**
    - **Triage Assessment:**
        - **Mechanism of Injury:**
        - **Red Flags:**
        - **Previous Treatment:**
        - **Functional Impact:**
    ---
3.  **Determine the Triage Pathway:** Based on the summary, choose ONE of the following pathways:
    - **Urgent Care / A&E:** For suspected fractures, dislocations, or severe neurological symptoms.
    - **GP / Primary Care:** For cases with potential red flags (fever, weight loss) or where the diagnosis is unclear and needs initial investigation.
    - **MSK Physiotherapy:** For mechanical pain, strains, or chronic issues that have NOT had prior physiotherapy. This is the most common pathway.
    - **SWLEOC Orthopaedic Surgery:** For patients with a clear mechanical problem who have ALREADY FAILED conservative treatment (e.g., extensive physiotherapy, GP management).
4.  **Provide a Justification:** After the pathway, add a "Justification:" section explaining your reasoning in 1-2 sentences.

**Conversation Transcript:**
{conversation_history}
"""

    async def summarize_and_triage(self, messages: List[Dict]) -> str:
        """Takes the full conversation and generates a final summary and triage recommendation."""
        conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        full_prompt = self.system_prompt_template.format(conversation_history=conversation_history)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                return ollama_response.get("response", "Could not generate summary.").strip()
        except Exception as e:
            print(f"Error during triage summary generation: {e}")
            return "Error: Could not generate the final triage summary."
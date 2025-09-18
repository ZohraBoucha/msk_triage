import httpx
from typing import List, Dict

class SummarizationAgent:
    """
    Analyzes a conversation transcript to produce a clinical summary and triage recommendation.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://localhost:11434/api/generate"
        
        # Prompt for initial clinical summary
        self.summary_prompt_template = """You are an Orthopaedic Triage Clinician. Analyze the conversation and provide a clinical summary.

FORMAT:
---
**Clinical Summary**
- **Presenting Complaint:**
- **SOCRATES:**
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
**Triage Pathway Recommendation:**
**Justification:**

**Conversation:**
{conversation_history}
"""

        # Prompt for differential diagnosis
        self.differential_prompt_template = """You are an Orthopaedic Triage Clinician. Based on the clinical summary below, provide a differential diagnosis.

FORMAT:
---
**Differential Diagnosis:**
- **Most Likely Diagnosis:** [Primary diagnosis based on symptoms]
- **Alternative Diagnoses:** [2-3 other possible diagnoses]
- **Red Flag Considerations:** [Serious conditions to rule out]
---

**Clinical Summary:**
{clinical_summary}
"""

    async def generate_differential_diagnosis(self, clinical_summary: str) -> str:
        """Generate differential diagnosis from clinical summary"""
        full_prompt = self.differential_prompt_template.format(clinical_summary=clinical_summary)
        
        try:
            print(f"DEBUG: Generating differential diagnosis...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                result = ollama_response.get("response", "Could not generate differential diagnosis.").strip()
                print(f"DEBUG: Differential diagnosis generated: {result[:100]}...")
                return result
        except Exception as e:
            print(f"Error during differential diagnosis generation: {e}")
            return "Error: Could not generate differential diagnosis."

    async def summarize_and_triage(self, messages: List[Dict]) -> str:
        """Takes the full conversation and generates a final summary and triage recommendation."""
        conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        full_prompt = self.summary_prompt_template.format(conversation_history=conversation_history)

        try:
            print(f"DEBUG: Generating clinical summary...")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                clinical_summary = ollama_response.get("response", "Could not generate summary.").strip()
                print(f"DEBUG: Clinical summary generated: {clinical_summary[:100]}...")
                
                # Now generate differential diagnosis from the clinical summary
                differential_diagnosis = await self.generate_differential_diagnosis(clinical_summary)
                
                # Combine both results
                full_summary = clinical_summary + "\n\n" + differential_diagnosis
                print(f"DEBUG: Full summary length: {len(full_summary)}")
                return full_summary
                
        except Exception as e:
            print(f"Error during triage summary generation: {e}")
            return "Error: Could not generate the final triage summary."
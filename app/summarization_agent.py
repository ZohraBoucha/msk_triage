import httpx
from typing import List, Dict, Any
from questionnaire_engine import run_questionnaire_engine, get_diagnosis_display_name
from questionnaire_specs import get_questionnaire_form
from triage_agent import TriageAgent

class SummarizationAgent:
    """
    Analyzes a conversation transcript to produce an SBAR clinical summary and differential diagnosis.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://localhost:11434/api/generate"
        
        # Prompt for SBAR clinical summary
        self.sbar_prompt_template = """You are an Orthopaedic Triage Clinician. Analyze the conversation and provide an SBAR clinical summary.

FORMAT:
---
**SITUATION:**
- **Patient Demographics:** [Age, gender, occupation if mentioned]
- **Presenting Complaint:** [Main problem in patient's words]
- **Body Part Affected:** [Specific anatomical location]

**BACKGROUND:**
- **Onset & Duration:** [When and how the problem started]
- **Mechanism of Injury:** [How it happened, if applicable]
- **Previous Treatment:** [What has been tried before]
- **Relevant History:** [Previous injuries, surgeries, comorbidities]

**ASSESSMENT:**
- **Clinical Findings:**
  - **Pain Characteristics:** [Type, location, radiation, severity]
  - **Associated Symptoms:** [Swelling, stiffness, weakness, etc.]
  - **Functional Impact:** [Effect on daily activities]
  - **Red Flags:** [Any concerning symptoms mentioned]
- **Physical Examination:** [Any exam findings mentioned]
- **Imaging:** [Any imaging results mentioned]

**RECOMMENDATION:**
- **Triage Pathway:** [Urgent/A&E, GP, MSK Physio, Orthopaedic Surgery]
- **Clinical Reasoning:** [Why this pathway is recommended]
- **Next Steps:** [Specific actions needed]

**Conversation:**
{conversation_history}
"""

        # Prompt for differential diagnosis using questionnaire engine
        self.differential_prompt_template = """You are an Orthopaedic Triage Clinician. Based on the clinical summary and questionnaire analysis below, provide a differential diagnosis.

FORMAT:
---
**DIFFERENTIAL DIAGNOSIS (Top 3):**

**1. PRIMARY DIAGNOSIS:**
- **Diagnosis:** [Most likely condition]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]
- **Score:** [Numerical score from questionnaire engine]

**2. SECONDARY DIAGNOSIS:**
- **Diagnosis:** [Second most likely condition]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]
- **Score:** [Numerical score from questionnaire engine]

**3. TERTIARY DIAGNOSIS:**
- **Diagnosis:** [Third most likely condition]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]
- **Score:** [Numerical score from questionnaire engine]

**RED FLAG CONSIDERATIONS:**
- [Any serious conditions that need to be ruled out]

**CLINICAL SUMMARY:**
{clinical_summary}

**QUESTIONNAIRE ANALYSIS:**
{questionnaire_analysis}
"""

    def _extract_patient_data_from_conversation(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract structured patient data from conversation for questionnaire analysis."""
        triage_agent = TriageAgent()
        return triage_agent._extract_patient_data(messages)

    def _run_questionnaire_analysis(self, patient_data: Dict[str, Any], questionnaire_type: str) -> Dict[str, Any]:
        """Run questionnaire analysis using the appropriate specification."""
        form = get_questionnaire_form(questionnaire_type)
        if not form:
            return {"error": "Questionnaire form not found"}
        
        spec = form["spec"]
        try:
            result = run_questionnaire_engine(spec, patient_data)
            return result
        except Exception as e:
            return {"error": f"Questionnaire analysis failed: {str(e)}"}

    async def generate_sbar_summary(self, messages: List[Dict]) -> str:
        """Generate SBAR clinical summary from conversation."""
        # Extract patient data for better demographics
        patient_data = self._extract_patient_data_from_conversation(messages)
        
        # Create enhanced conversation with patient data
        conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        
        # Add patient data context to the prompt
        patient_context = ""
        if patient_data.get("patient", {}).get("age_years"):
            patient_context += f"Patient Age: {patient_data['patient']['age_years']} years. "
        if patient_data.get("patient", {}).get("gender"):
            patient_context += f"Patient Gender: {patient_data['patient']['gender']}. "
        
        enhanced_conversation = f"PATIENT CONTEXT: {patient_context}\n\n{conversation_history}"
        full_prompt = self.sbar_prompt_template.format(conversation_history=enhanced_conversation)

        try:
            print(f"DEBUG: Generating SBAR summary...")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                sbar_summary = ollama_response.get("response", "Could not generate SBAR summary.").strip()
                print(f"DEBUG: SBAR summary generated: {sbar_summary[:100]}...")
                return sbar_summary
        except Exception as e:
            print(f"Error during SBAR summary generation: {e}")
            return "Error: Could not generate SBAR summary."

    async def generate_differential_diagnosis(self, clinical_summary: str, questionnaire_analysis: Dict[str, Any]) -> str:
        """Generate differential diagnosis using questionnaire analysis results."""
        # Format questionnaire analysis for the prompt
        analysis_text = ""
        if "error" in questionnaire_analysis:
            analysis_text = f"Questionnaire analysis error: {questionnaire_analysis['error']}"
        else:
            if questionnaire_analysis.get("route") == "urgent":
                analysis_text = f"URGENT ASSESSMENT REQUIRED\nReason: {questionnaire_analysis.get('urgent_reason', 'Unknown')}\nProvisional Diagnosis: {questionnaire_analysis.get('provisional_diagnosis', 'Unknown')}"
            else:
                top_diagnoses = questionnaire_analysis.get("top", [])
                analysis_text = "QUESTIONNAIRE ANALYSIS RESULTS:\n"
                for i, dx in enumerate(top_diagnoses, 1):
                    diagnosis_name = get_diagnosis_display_name(dx.get("diagnosis_code", "Unknown"))
                    analysis_text += f"{i}. {diagnosis_name} (Score: {dx.get('score', 0)}, Confidence: {dx.get('confidence_band', 'Unknown')})\n"
                    analysis_text += f"   Key Features: {', '.join(dx.get('key_drivers', []))}\n"
                
                safety_net = questionnaire_analysis.get("safety_net", [])
                if safety_net:
                    analysis_text += f"\nSafety Considerations: {'; '.join(safety_net)}"

        full_prompt = self.differential_prompt_template.format(
            clinical_summary=clinical_summary,
            questionnaire_analysis=analysis_text
        )
        
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
        """Takes the full conversation and generates an SBAR summary with differential diagnosis."""
        try:
            # Extract patient data for questionnaire analysis
            patient_data = self._extract_patient_data_from_conversation(messages)
            
            # Determine questionnaire type (simplified - could be enhanced)
            questionnaire_type = "knee_oa"  # Default, could be determined from conversation
            if any(word in str(messages).lower() for word in ['injury', 'hurt', 'injured', 'accident']):
                questionnaire_type = "knee_injury"
            
            # Run questionnaire analysis
            questionnaire_analysis = self._run_questionnaire_analysis(patient_data, questionnaire_type)
            
            # Generate SBAR summary
            sbar_summary = await self.generate_sbar_summary(messages)
            
            # Generate differential diagnosis
            differential_diagnosis = await self.generate_differential_diagnosis(sbar_summary, questionnaire_analysis)
            
            # Combine results
            full_summary = sbar_summary + "\n\n" + differential_diagnosis
            print(f"DEBUG: Full summary length: {len(full_summary)}")
            return full_summary
                
        except Exception as e:
            print(f"Error during triage summary generation: {e}")
            return "Error: Could not generate the final triage summary."
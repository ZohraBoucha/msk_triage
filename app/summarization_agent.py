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

CRITICAL RULES:
- Record patient name, age, and gender exactly as stated - do not editorialize or correct
- Use clinical language in SBAR (e.g., "subjective instability" not "wobbly")
- Classify locking explicitly: "true locking" (won't move, needs manoeuvre) vs "pseudo-locking/catching" (pops and goes)
- Always include quantified functional metrics: max walking distance, stairs tolerance, night/rest pain, work impact, falls
- For imaging: specify modality, structure, side, date; if vague, note "report not available"
- Keep recommendations tight: Pathway → Reason → Next step
- Never add editorial comments about name/gender mismatches in the clinical summary

FORMAT:
---
**SITUATION:**
- **Patient Demographics:** [Age, stated gender, occupation if mentioned]
- **Presenting Complaint:** [Main problem in patient's words]
- **Body Part Affected:** [Specific anatomical location]

**BACKGROUND:**
- **Onset & Duration:** [When and how the problem started]
- **Mechanism of Injury:** [How it happened, if applicable]
- **Previous Treatment:** [What has been tried before - include duration and response]
- **Relevant History:** [Previous injuries, surgeries, comorbidities]

**ASSESSMENT:**
- **Clinical Findings:**
  - **Pain Characteristics:** [Type, location, radiation, severity]
  - **Associated Symptoms:** [Swelling, stiffness, weakness, etc.]
  - **Functional Impact:** [Quantified metrics: max walking distance, stairs tolerance, night/rest pain Y/N, work adjustments, falls Y/N, aid use]
  - **Instability/Locking:** [Classify: true locking vs pseudo-locking/catching, subjective instability type]
  - **Red Flags:** [Any concerning symptoms mentioned]
- **Physical Examination:** [Any exam findings mentioned]
- **Imaging:** [Modality, structure, side, date - if vague, note "report not available"]

**RECOMMENDATION:**
- **Pathway:** [Specific clinic/service]
- **Reason:** [Why this pathway - failed conservative care, imaging findings, functional impact]
- **Next Step:** [Specific action - book consult, bring imaging, consider specific exam]

**SAFETY NET:**
- **Urgent Care:** Seek urgent care if severe, rapidly worsening pain/swelling, fever, new neurological symptoms, bladder/bowel dysfunction, new calf swelling/shortness of breath, or new frank giving-way with falls.
- **Follow-up:** [Specific follow-up instructions based on condition]

**Conversation:**
{conversation_history}
"""

        # Prompt for differential diagnosis using questionnaire engine
        self.differential_prompt_template = """You are an Orthopaedic Triage Clinician. Based on the clinical summary below, provide a differential diagnosis.

FORMAT:
---
**DIFFERENTIAL DIAGNOSIS (Top 3):**

**1. PRIMARY DIAGNOSIS:**
- **Diagnosis:** [Most likely condition based on clinical presentation]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]

**2. SECONDARY DIAGNOSIS:**
- **Diagnosis:** [Second most likely condition]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]

**3. TERTIARY DIAGNOSIS:**
- **Diagnosis:** [Third most likely condition]
- **Confidence:** [High/Moderate/Low]
- **Key Supporting Features:** [Main clinical features supporting this diagnosis]

**RED FLAG CONSIDERATIONS:**
- [Any serious conditions that need to be ruled out]

**SAFETY NET:**
- **Urgent Care:** If symptoms worsen suddenly, or new red flags occur (severe weakness, fever, bladder/bowel problems, severe pain), seek urgent care immediately.
- **Follow-up:** [Specific follow-up instructions based on condition]

**CLINICAL SUMMARY:**
{clinical_summary}
"""

        # Prompt for soft tissue vs arthroplasty triage classification
        self.triage_classification_prompt_template = """You are an Orthopaedic Triage Clinician. Based on the clinical summary below, classify this case into the appropriate triage category.

CLASSIFICATION RULES:
- **Soft Tissue**: Cases involving ligaments, tendons, muscles, cartilage, or other soft tissue structures that do not require joint replacement. This includes ALL ligament injuries (ACL, PCL, MCL, LCL, MPFL), meniscal tears, tendon injuries, muscle strains, and cartilage injuries.
- **Arthroplasty**: Cases involving severe joint degeneration, end-stage arthritis, or conditions that may require joint replacement surgery

EXAMPLES:
- Soft Tissue: ACL tears, PCL tears, MCL tears, LCL tears, MPFL tears, meniscal tears, rotator cuff tears, tendonitis, muscle strains, ligament sprains, cartilage injuries, bursitis
- Arthroplasty: End-stage osteoarthritis, severe joint degeneration, failed conservative treatment for arthritis, joint replacement candidates

IMPORTANT: Any ligament injury (ACL, PCL, MCL, LCL, MPFL) or meniscal tear is ALWAYS Soft Tissue, never Arthroplasty.

FORMAT:
---
**TRIAGE CLASSIFICATION:**

**Category:** [Soft Tissue / Arthroplasty]
**Body Part:** [Specific anatomical location, e.g., "Knee", "Shoulder", "Hip"]
**Specialty:** [Soft Tissue - Knee / Knee Arthroplasty]
**Clinical Reasoning:** [Brief explanation of why this classification was chosen]

**CLINICAL SUMMARY:**
{clinical_summary}
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

    def _enhance_imaging_specificity(self, imaging_history: str) -> str:
        """Enhance imaging specificity by asking for clarification if vague."""
        if not imaging_history or imaging_history.lower() in ["none", "no imaging", "not mentioned"]:
            return "No imaging mentioned"
        
        # Check if imaging details are vague
        vague_terms = ["torn ligament", "ligament issue", "some damage", "problems", "issues"]
        if any(term in imaging_history.lower() for term in vague_terms):
            # Try to extract what we can and note what's missing
            if "mri" in imaging_history.lower():
                return f"MRI (date unknown) reportedly shows {imaging_history} - specific ligament and grade not specified in conversation"
            else:
                return f"Imaging reportedly shows {imaging_history} - specific details not available"
        
        return imaging_history

    def _apply_triage_guardrails(self, patient_data: Dict[str, Any], conversation_text: str) -> str:
        """
        Returns one of:
          - 'urgent_ed'
          - 'orthopaedic_soft_tissue'
          - 'arthroplasty'
          - 'msk_physio'
          - 'gp_primary'
        """
        import re
        age = int(patient_data.get("patient", {}).get("age_years", 0))
        sx   = (patient_data.get("symptoms") or "").lower()
        fx   = (patient_data.get("functional_impact") or "").lower()
        img  = (patient_data.get("imaging_history") or "").lower()
        tx   = (patient_data.get("previous_treatment") or "").lower()
        convo= (conversation_text or "").lower()

        text = " ".join([sx, fx, img, tx, convo])

        def has(pattern):
            return re.search(pattern, text) is not None

        def has_negated(term):
            # e.g., "no true locking", "denies fever"
            return re.search(rf"(no|den(y|ies)|without)\s+\b{term}\b", text) is not None

        def present(term):
            # present only if not explicitly negated
            return (term in text) and not has_negated(term)

        # ---------- URGENT FLAGS ----------
        urgent = 0
        # septic arthritis / infection
        if any(present(t) for t in ["fever", "rigors", "chills", "hot swollen joint", "erythema", "sepsis", "septic"]):
            urgent += 3
        # fracture / dislocation / unable to weight-bear after trauma
        if any(present(t) for t in ["deformity", "audible crack", "unable to weight-bear", "dislocation"]) or has(r"\bfracture\b"):
            urgent += 3
        # neurovascular
        if any(present(t) for t in ["numbness", "foot drop", "pins and needles", "cold foot", "pale foot", "weak pulse"]):
            urgent += 2
        # DVT/PE risk
        if any(present(t) for t in ["calf swelling", "calf tenderness", "sudden breathlessness", "pleuritic chest pain"]):
            urgent += 2
        # cancer red flags
        if any(present(t) for t in ["unexplained weight loss", "night sweats", "history of cancer"]) and any(present(t) for t in ["night pain", "rest pain"]):
            urgent += 2
        if urgent >= 3:
            return "urgent_ed"

        # ---------- SOFT-TISSUE ORTHO ----------
        soft_tissue = 0
        # instability / giving way / dislocation / patellar instability
        if any(present(t) for t in ["instability", "giving way", "dislocation", "pops out", "kneecap out", "patellar instability"]):
            soft_tissue += 3
        # true mechanical block
        if (present("true locking") or present("won't move") or present("completely stuck")) and not present("no true locking"):
            soft_tissue += 3
        # traumatic mechanism with persistent symptoms >6–12 weeks
        if any(present(t) for t in ["pivot", "twist", "dashboard", "skiing", "tackle", "contact injury"]):
            soft_tissue += 2
        if any(present(t) for t in ["acl", "pcl", "mcl", "lcl", "mpfl", "meniscal tear", "bucket handle", "rupture", "torn ligament", "ligament tear", "posterolateral corner"]):
            soft_tissue += 3
        if any(present(t) for t in ["failed physio", "failed physiotherapy", "completed 12 weeks physio", "persistent despite rehab"]):
            soft_tissue += 2
        # age bias (younger patients more likely soft tissue pathway)
        if age < 50:
            soft_tissue += 1

        # ---------- ARTHROPLASTY ----------
        arthro = 0
        # radiographic OA markers
        if any(present(t) for t in [
            "kellgren", "joint space narrowing", "osteophytes", "tricompartmental oa",
            "bone-on-bone", "end-stage", "severe degenerative osteoarthritis", "advanced oa"
        ]) or ("osteoarthritis" in text and any(present(t) for t in ["severe", "advanced", "end-stage"])):
            arthro += 3
        # age and severity
        if age >= 55:
            arthro += 1
        # functional collapse
        if any(present(t) for t in ["daily function severely limited", "unable to manage stairs", "housebound", "walking distance < 200m", "needs two sticks"]):
            arthro += 2
        # persistent night/rest pain most nights
        if present("night pain") or present("rest pain"):
            arthro += 1
        # failed non-op incl. injections / multiple physio rounds
        if any(present(t) for t in ["failed conservative", "failed non-operative", "steroid injection with short-lived relief", "multiple courses of physio"]):
            arthro += 1

        # ---------- MSK PHYSIO ----------
        physio = 0
        if any(present(t) for t in ["patellofemoral pain", "pfps", "chondromalacia", "iliotibial band", "itbs", "tendinopathy", "pes anserine", "bursitis"]):
            physio += 2
        if any(present(t) for t in ["degenerative meniscal tear", "meniscal signal", "small tear"]) and present("no true locking"):
            physio += 2
        if any(present(t) for t in ["mild symptoms", "manageable", "can still work", "no instability", "no giving way"]):
            physio += 1
        # short duration or early rehab
        if any(present(t) for t in ["< 12 weeks", "six weeks", "8 weeks"]) or any(present(t) for t in ["early rehab", "starting physio", "conservative management"]):
            physio += 1
        if age < 55 and not any(present(t) for t in ["night pain", "rest pain"]):
            physio += 1

        # ---------- GP / PRIMARY ----------
        gp = 0
        if "osteoarthritis" in text and arthro < 3:
            gp += 2  # likely OA management optimization rather than surgery
        if any(present(t) for t in ["analgesia review", "weight loss", "activity modification", "home exercise", "injection discussion"]):
            gp += 1
        if not any(present(t) for t in ["instability", "giving way", "true locking"]) and physio == 0:
            gp += 1

        # ---------- PICK PATHWAY WITH PRIORITY ----------
        # Priority order: urgent_ed > soft_tissue > arthroplasty > msk_physio > gp_primary
        scores = {
            "orthopaedic_soft_tissue": soft_tissue,
            "arthroplasty": arthro,
            "msk_physio": physio,
            "gp_primary": gp,
        }

        # If any strong soft-tissue signal, prefer that over arthro if age <55 and no OA imaging
        if soft_tissue >= 4 and not ("advanced" in text or "end-stage" in text or "bone-on-bone" in text):
            best = "orthopaedic_soft_tissue"
        else:
            best = max(scores, key=scores.get)

        # Safe default if everything is low-signal
        if all(v == 0 for v in scores.values()):
            return "msk_physio"

        return best

    async def generate_sbar_summary(self, messages: List[Dict]) -> str:
        """Generate SBAR clinical summary from conversation."""
        # Extract patient data for better demographics
        patient_data = self._extract_patient_data_from_conversation(messages)
        
        # Apply triage guardrails
        triage_context = self._apply_triage_guardrails(patient_data, str(messages))
        
        # Enhance imaging specificity
        imaging_context = self._enhance_imaging_specificity(patient_data.get("imaging_history", ""))
        
        # Format conversation history
        conversation_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
        
        # Create the full prompt
        full_prompt = self.sbar_prompt_template.format(
            conversation_history=conversation_history
        )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                result = ollama_response.get("response", "Could not generate SBAR summary.").strip()
                return result
        except Exception as e:
            print(f"Error during SBAR summary generation: {e}")
            print(f"Error type: {type(e)}")
            return "Error: Could not generate SBAR summary."

    async def generate_differential_diagnosis(self, clinical_summary: str) -> str:
        """Generate differential diagnosis from clinical summary."""
        full_prompt = self.differential_prompt_template.format(
            clinical_summary=clinical_summary
        )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                result = ollama_response.get("response", "Could not generate differential diagnosis.").strip()
                return result
        except Exception as e:
            print(f"Error during differential diagnosis generation: {e}")
            print(f"Error type: {type(e)}")
            return "Error: Could not generate differential diagnosis."

    async def generate_triage_classification(self, clinical_summary: str) -> str:
        """Generate soft tissue vs arthroplasty triage classification."""
        full_prompt = self.triage_classification_prompt_template.format(
            clinical_summary=clinical_summary
        )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                result = ollama_response.get("response", "Could not generate triage classification.").strip()
                return result
        except Exception as e:
            print(f"Error during triage classification generation: {e}")
            print(f"Error type: {type(e)}")
            return "Error: Could not generate triage classification."

    async def summarize_and_triage(self, messages: List[Dict]) -> str:
        """Generate complete clinical summary with SBAR, differential diagnosis, and triage classification."""
        # Generate SBAR summary
        sbar_summary = await self.generate_sbar_summary(messages)
        
        # Generate differential diagnosis
        differential_diagnosis = await self.generate_differential_diagnosis(sbar_summary)
        
        # Generate triage classification
        triage_classification = await self.generate_triage_classification(sbar_summary)
        
        # Combine all results
        full_summary = f"{sbar_summary}\n\n{differential_diagnosis}\n\n{triage_classification}"
        
        return full_summary

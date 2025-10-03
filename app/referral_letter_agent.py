import httpx
from typing import List, Dict, Any
from .questionnaire_engine import run_questionnaire_engine, get_diagnosis_display_name
from .questionnaire_specs import get_questionnaire_form
from .triage_agent import TriageAgent

class ReferralLetterAgent:
    """
    Generates detailed referral letters for various specialties (SWLEOC, Physio, GP, etc.)
    based on clinical summaries and triage decisions.
    """
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.ollama_api_url = "http://localhost:11434/api/generate"
        
        # Prompt for SWLEOC referral letter
        self.swleoc_referral_prompt_template = """You are an Orthopaedic Triage Clinician writing a detailed referral letter to SWLEOC (South West London Elective Orthopaedic Centre).

Based on the clinical summary below, generate a comprehensive referral letter in proper letter format that includes all necessary medical information for the receiving orthopaedic team.

Write this as a formal medical letter with proper salutation, body paragraphs, and closing. Use flowing prose rather than bullet points, but ensure all critical information is included.

LETTER FORMAT:
---
Dear Colleague,

I am writing to refer [Patient name] for orthopaedic assessment and management.

**Patient Details:**
[Patient name], [age] year old [gender], [DOB if available], NHS Number [if available], under the care of [GP name and practice if available].

**Presenting Complaint:**
[Main problem in patient's own words] affecting the [specific anatomical location]. The problem began [when and how] and has been present for [duration].

**History of Presenting Complaint:**
The patient reports [detailed description of onset, progression, pain characteristics, associated symptoms]. Specifically, [pain description with location, severity, aggravating/relieving factors]. Associated symptoms include [swelling, stiffness, weakness, instability, locking, etc.].

Functional impact is significant with [specific quantified metrics: walking distance, stairs tolerance, work impact, ADLs, night/rest pain, falls, aid use].

**Past Medical History:**
Previous injuries include [previous injuries to the same or other joints]. Previous surgery: [any relevant surgical history]. Comorbidities: [relevant medical conditions]. Current medications: [current medications, especially pain management]. Allergies: [known allergies].

**Treatment to Date:**
The patient has undergone [conservative management details] for [duration]. Response to treatment has been [effectiveness of previous treatments]. Current analgesia includes [current pain management regimen].

**Clinical Examination Findings:**
On examination, [inspection findings - swelling, deformity, muscle wasting]. Palpation reveals [tenderness, warmth, effusion]. Range of motion shows [active and passive ROM measurements if available]. Special tests: [specific orthopaedic tests performed and results]. Neurovascular status: [distal neurovascular examination]. Gait: [walking pattern if relevant].

**Investigations:**
Imaging includes [X-rays, MRI, CT, ultrasound with dates and findings]. Laboratory tests: [blood tests, inflammatory markers if relevant]. Other investigations: [any other relevant tests].

**Assessment:**
The primary diagnosis is [most likely condition with confidence level]. Secondary considerations include [other possible conditions]. Red flag conditions have been considered and ruled out: [serious conditions that have been considered and excluded].

**Clinical Reasoning:**
This patient requires orthopaedic review because [specific clinical reasoning]. Conservative management has failed as evidenced by [evidence of failed non-operative treatment]. The functional impact on quality of life and work is [impact description]. Surgical considerations may include [potential surgical options if applicable].

**Patient Expectations:**
The patient's main concerns are [what the patient is most worried about]. Their functional goals include [what the patient hopes to achieve]. Work/sport requirements: [specific functional demands].

**Urgency:**
This referral is [Routine/Urgent/2-week wait] because [reasoning for urgency level].

**Specific Request:**
I would be grateful if you could [specific consultation, imaging, procedure]. Specific questions for your consideration: [specific clinical questions]. Follow-up arrangements: [how follow-up should be managed].

**Safety Net:**
The patient has been advised to seek urgent care if [red flag symptoms]. Follow-up instructions: [what the patient should do while waiting].

**Conclusion:**
[Patient name] is a [age] year old [gender] with [brief summary of condition]. Despite [conservative treatment], symptoms persist with significant functional impact. I believe this patient would benefit from your specialist assessment and management.

I look forward to your assessment and recommendations.

Yours sincerely,

[Your Name]
[Your Title]
MSK Triage Service
[Contact details]
[Date]

**CLINICAL SUMMARY:**
{clinical_summary}

**TRIAGE DECISION:**
{triage_decision}

**CONVERSATION EXCERPT:**
{conversation_excerpt}
"""

        # Prompt for Physiotherapy referral letter
        self.physio_referral_prompt_template = """You are an Orthopaedic Triage Clinician writing a detailed referral letter to MSK Physiotherapy.

Based on the clinical summary below, generate a comprehensive referral letter in proper letter format that includes all necessary information for the physiotherapy team.

Write this as a formal medical letter with proper salutation, body paragraphs, and closing. Use flowing prose rather than bullet points, but ensure all critical information is included.

LETTER FORMAT:
---
Dear Physiotherapy Team,

I am writing to refer [Patient name] for physiotherapy assessment and management.

**Patient Details:**
[Patient name], [age] year old [gender], [contact details if available].

**Presenting Complaint:**
[Main problem] affecting the [specific anatomical location]. The problem began [how the problem started] and has been present for [duration].

**Pain Assessment:**
The patient reports [exact location of pain] with [type of pain - sharp, dull, aching, burning, etc.]. Pain severity is [0-10 scale, worst and current]. Pain is aggravated by [what makes it worse] and relieved by [what helps]. The pain pattern is [constant, intermittent, morning stiffness, etc.].

**Functional Impact:**
Functional limitations include [walking distance, aids needed, stairs]. Work impact: [impact on work duties]. Activities of daily living affected: [specific ADLs]. Sport/exercise impact: [impact on recreational activities]. Sleep quality: [impact on sleep quality].

**Current Functional Status:**
Mobility: [walking aids, assistance needed]. Range of motion shows [current limitations]. Strength assessment reveals [muscle weakness areas]. Balance: [balance issues if relevant].

**Treatment History:**
Previous physiotherapy: [when, where, what was done, response]. Other treatments include [medications, injections, other therapies]. Self-management attempts: [what patient has tried at home].

**Clinical Findings:**
Posture: [postural issues observed]. Movement patterns show [compensatory movements, limping]. Muscle imbalances noted: [weak or tight muscle groups]. Special tests: [relevant physiotherapy assessment findings].

**Goals for Physiotherapy:**
Primary goals include [main functional goals]. Pain management targets: [pain reduction targets]. Functional goals: [specific functional improvements]. Return to activity goals: [sport, work, recreational goals].

**Treatment Recommendations:**
Suggested approach: [manual therapy, exercise, education, etc.]. Recommended frequency: [suggested treatment frequency]. Expected duration: [expected treatment duration]. Home exercise program should focus on [specific exercises to focus on].

**Contraindications/Precautions:**
Safety considerations: [any precautions needed]. Red flags to watch for: [signs to watch for]. Medical considerations: [relevant medical factors].

**Conclusion:**
[Patient name] is a [age] year old [gender] with [brief summary of condition]. I believe this patient would benefit from your specialist physiotherapy assessment and management to address [specific functional goals].

I look forward to your assessment and treatment recommendations.

Yours sincerely,

[Your Name]
[Your Title]
MSK Triage Service
[Date]

**CLINICAL SUMMARY:**
{clinical_summary}

**TRIAGE DECISION:**
{triage_decision}
"""

        # Prompt for GP referral letter
        self.gp_referral_prompt_template = """You are an Orthopaedic Triage Clinician writing a detailed referral letter to the patient's GP.

Based on the clinical summary below, generate a comprehensive referral letter in proper letter format for ongoing primary care management.

Write this as a formal medical letter with proper salutation, body paragraphs, and closing. Use flowing prose rather than bullet points, but ensure all critical information is included.

LETTER FORMAT:
---
Dear Dr [GP Name],

I am writing to provide you with an update on [Patient name] following their MSK triage assessment.

**Patient Details:**
[Patient name], [age] year old [gender], NHS Number [if available].

**Clinical Presentation:**
[Patient name] presented with [main problem] affecting the [specific anatomical location]. The problem has been present for [duration].

**Current Symptoms:**
The patient reports [location, severity, pattern of pain]. Functional impact includes [impact on daily activities]. Associated symptoms include [swelling, stiffness, weakness].

**Assessment:**
Clinical findings include [key examination findings]. The most likely diagnosis is [most likely conditions]. Red flag conditions have been considered and ruled out: [serious conditions considered and excluded].

**Treatment Recommendations:**
For immediate management, I recommend [pain relief, activity modification]. Please review their current analgesia and consider [potential adjustments]. Lifestyle modifications should include [weight management, activity advice]. Self-management strategies include [home exercises, activity pacing].

**Follow-up Arrangements:**
I suggest reviewing progress in [when to review progress]. Please consider re-referral to specialist services if [indicators for re-referral]. The patient has been educated about red flag symptoms including [warning signs to watch for].

**Patient Education:**
The patient has been provided with information about [what the patient should understand about their condition]. Expected recovery timeline: [expected recovery timeline]. Self-management strategies include [what patient can do at home].

**Conclusion:**
[Patient name] is a [age] year old [gender] with [brief summary of condition]. I believe this patient can be managed in primary care with the above recommendations. Please do not hesitate to contact me if you have any questions or if the patient's condition changes.

Yours sincerely,

[Your Name]
[Your Title]
MSK Triage Service
[Date]

**CLINICAL SUMMARY:**
{clinical_summary}

**TRIAGE DECISION:**
{triage_decision}
"""

    def _extract_patient_data_from_conversation(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract structured patient data from conversation for referral analysis."""
        triage_agent = TriageAgent()
        return triage_agent._extract_patient_data(messages)

    def _determine_referral_type(self, triage_decision: str, clinical_summary: str) -> str:
        """Determine the most appropriate referral type based on triage decision and clinical summary."""
        triage_lower = triage_decision.lower()
        summary_lower = clinical_summary.lower()
        
        # Check for urgent/emergency cases
        if any(term in triage_lower for term in ["urgent", "emergency", "ed", "a&e"]):
            return "urgent_ed"
        
        # Check for orthopaedic surgery cases
        if any(term in triage_lower for term in ["orthopaedic", "soft tissue", "arthroplasty", "surgery", "swleoc"]):
            return "swleoc"
        
        # Check for physiotherapy cases
        if any(term in triage_lower for term in ["physio", "physiotherapy", "msk physio", "conservative"]):
            return "physio"
        
        # Check for GP management cases
        if any(term in triage_lower for term in ["gp", "primary", "general practice"]):
            return "gp"
        
        # Default based on clinical summary content
        if any(term in summary_lower for term in ["surgery", "operative", "arthroplasty", "replacement"]):
            return "swleoc"
        elif any(term in summary_lower for term in ["physio", "exercise", "conservative", "rehabilitation"]):
            return "physio"
        else:
            return "gp"

    async def generate_referral_letter(self, clinical_summary: str, triage_decision: str, 
                                     conversation_messages: List[Dict], referral_type: str = None) -> str:
        """Generate a detailed referral letter based on the clinical summary and triage decision."""
        
        # Determine referral type if not specified
        if not referral_type:
            referral_type = self._determine_referral_type(triage_decision, clinical_summary)
        
        # Extract patient data for additional context
        patient_data = self._extract_patient_data_from_conversation(conversation_messages)
        
        # Create conversation excerpt (last 10 messages for context)
        conversation_excerpt = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation_messages[-10:]
        ])
        
        # Select appropriate prompt template
        if referral_type == "swleoc":
            prompt_template = self.swleoc_referral_prompt_template
        elif referral_type == "physio":
            prompt_template = self.physio_referral_prompt_template
        elif referral_type == "gp":
            prompt_template = self.gp_referral_prompt_template
        else:
            # Default to SWLEOC for complex cases
            prompt_template = self.swleoc_referral_prompt_template
        
        # Format the prompt
        full_prompt = prompt_template.format(
            clinical_summary=clinical_summary,
            triage_decision=triage_decision,
            conversation_excerpt=conversation_excerpt
        )
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.ollama_api_url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                response.raise_for_status()
                ollama_response = response.json()
                result = ollama_response.get("response", "Could not generate referral letter.").strip()
                return result
        except Exception as e:
            print(f"Error during referral letter generation: {e}")
            print(f"Error type: {type(e)}")
            return f"Error: Could not generate referral letter. Error: {str(e)}"

    async def generate_all_referral_letters(self, clinical_summary: str, triage_decision: str, 
                                          conversation_messages: List[Dict]) -> Dict[str, str]:
        """Generate referral letters for all appropriate specialties."""
        referrals = {}
        
        # Determine primary referral type
        primary_type = self._determine_referral_type(triage_decision, clinical_summary)
        
        # Generate primary referral
        referrals[primary_type] = await self.generate_referral_letter(
            clinical_summary, triage_decision, conversation_messages, primary_type
        )
        
        # Generate additional referrals if appropriate
        if primary_type == "swleoc":
            # Also generate physio referral for pre/post-op care
            referrals["physio"] = await self.generate_referral_letter(
                clinical_summary, triage_decision, conversation_messages, "physio"
            )
        elif primary_type == "physio":
            # Also generate GP referral for ongoing management
            referrals["gp"] = await self.generate_referral_letter(
                clinical_summary, triage_decision, conversation_messages, "gp"
            )
        
        return referrals

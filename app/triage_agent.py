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
    
    # SWLEOC Eligibility Questions
    GATHER_DETAILED_TREATMENT_HISTORY = "GATHER_DETAILED_TREATMENT_HISTORY"
    GATHER_SURGERY_INTEREST = "GATHER_SURGERY_INTEREST"
    GATHER_CONSERVATIVE_TREATMENT_FAILURE = "GATHER_CONSERVATIVE_TREATMENT_FAILURE"
    
    # Additional Comprehensive Questions
    GATHER_SMOKING_STATUS = "GATHER_SMOKING_STATUS"
    GATHER_PREVIOUS_INJURY_SURGERY = "GATHER_PREVIOUS_INJURY_SURGERY"
    GATHER_TREATMENT_RESPONSE = "GATHER_TREATMENT_RESPONSE"
    GATHER_LOCKING_TYPE = "GATHER_LOCKING_TYPE"
    GATHER_OVERUSE_CONTEXT = "GATHER_OVERUSE_CONTEXT"
    GATHER_OA_INDEX_DETAILED = "GATHER_OA_INDEX_DETAILED"
    GATHER_IMAGING_HISTORY = "GATHER_IMAGING_HISTORY"
    GATHER_PHENOTYPE_SYMPTOMS = "GATHER_PHENOTYPE_SYMPTOMS"
    
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
        self.question_count = {}  # Track how many times each question has been asked
        
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
            
            # --- Basic Demographics ---
            TriageState.GATHER_AGE: "What is your age?",
            TriageState.GATHER_LATERALITY: "Which side is affected - left or right?",
            
            # --- Onset and Mechanism ---
            TriageState.GATHER_DURATION: "How long have you had this problem? Is it acute (less than 2 weeks), subacute (2-12 weeks), or chronic (more than 12 weeks)?",
            TriageState.GATHER_MECHANISM: "How did this problem start? Was it from an injury, overuse, or did it come on gradually?",
            
            # --- Symptoms ---
            TriageState.GATHER_SYMPTOMS: "Can you describe your main symptoms in your own words?",
            TriageState.GATHER_PAIN_CHARACTER: "What does the pain feel like? Is it sharp, dull, aching, burning, or something else?",
            TriageState.GATHER_RADIATION: "Does the pain or feeling spread to any other part of your body?",
            TriageState.GATHER_ASSOCIATED_SYMPTOMS: "Do you have any other symptoms like swelling, stiffness, numbness, or weakness?",
            TriageState.GATHER_TIMING: "Are your symptoms constant or do they come and go? Is there any pattern to them?",
            TriageState.GATHER_EXACERBATING_RELIEVING: "What makes your symptoms better or worse?",
            TriageState.GATHER_SEVERITY: "On a scale of 0 to 10, where 0 is no pain and 10 is the worst imaginable, how would you rate your pain?",
            
            # --- Stiffness (for OA) ---
            TriageState.GATHER_STIFFNESS: "Do you experience morning stiffness? If so, how long does it take to loosen up in the morning?",
            
            # --- Functional Impact ---
            TriageState.GATHER_FUNCTIONAL_IMPACT: "How is this problem affecting your daily activities, work, or hobbies?",
            TriageState.GATHER_PREVIOUS_TREATMENT: "What treatments have you tried before? For example, medications, physiotherapy, or other interventions?",
            TriageState.GATHER_RED_FLAGS: "To make sure we're not missing anything serious, have you experienced any fever, chills, unexplained weight loss, or severe weakness?",
            
            # --- Specialized Questions (Knee Injury) ---
            TriageState.GATHER_KNEE_SCORE: "How would you rate your knee function on various activities like walking, stairs, or squatting?",
            TriageState.GATHER_EXAM_FINDINGS: "Do you know about any physical examination findings? For example, swelling, instability, or other signs?",
            TriageState.GATHER_IMAGING: "Have you had any imaging done like X-rays or MRI? If so, what were the results?",
            
            # --- SWLEOC Eligibility Questions ---
            TriageState.GATHER_SURGERY_INTEREST: "Given the nature of your condition, would you be interested in considering surgical treatment if it was recommended by a specialist?",
            TriageState.GATHER_CONSERVATIVE_TREATMENT_FAILURE: "Have you tried conservative treatments like physiotherapy, injections, or other non-surgical approaches, and if so, did they help with your symptoms?",
            
            # --- Additional Comprehensive Questions ---
            TriageState.GATHER_SMOKING_STATUS: "Do you smoke cigarettes or use any tobacco products?",
            TriageState.GATHER_PREVIOUS_INJURY_SURGERY: "Have you had any previous injuries or surgeries to this area? For example, ACL reconstruction, meniscus surgery, knee replacement, or arthroscopy?",
            TriageState.GATHER_TREATMENT_RESPONSE: "When you tried the treatments we discussed, did they help with your symptoms, make no difference, or make things worse?",
            TriageState.GATHER_LOCKING_TYPE: "When you experience locking or catching, is it more like the joint gets completely stuck and won't move (true locking), or is it more like a brief click or catch that pops and goes?",
            TriageState.GATHER_OVERUSE_CONTEXT: "Is this related to any specific activities like running, marathon training, increased mileage, hill repeats, or prolonged standing?",
            TriageState.GATHER_OA_INDEX_DETAILED: "Let me ask about specific activities. How much difficulty do you have with: going down stairs, going up stairs, rising from a chair, bending to the floor, getting in/out of a car, putting on socks, getting in/out of the bath, and heavy domestic work?",
            TriageState.GATHER_IMAGING_HISTORY: "Have you had any imaging done like X-rays or MRI scans? If so, what were the results?",
            TriageState.GATHER_PHENOTYPE_SYMPTOMS: "Do you experience any of the following: joint instability or giving way, locking or catching sensations, or pain specifically at the front of the knee or behind the kneecap?",
            
            # --- Completion ---
            TriageState.COMPLETE: "Thank the user for all the information. State that you now have a complete picture of their situation and that a clinical summary with differential diagnosis will be prepared for the clinical team to direct them to the most appropriate care pathway."
        }
        return prompts.get(state, "The conversation is complete.")

    def _determine_current_state(self, messages: List[Dict]) -> TriageState:
        """Determines the current state based on the conversation history and questionnaire type."""
        # Extract patient data to check what information we already have
        patient_data = self._extract_patient_data(messages)
        
        # Determine questionnaire type based on conversation
        if not self.current_questionnaire:
            # Look for body part and injury type in conversation
            last_user_message = messages[-1]['content'].lower() if messages else ""
            
            if 'knee' in last_user_message:
                if any(word in last_user_message for word in ['injury', 'hurt', 'injured', 'accident', 'fall', 'twist']):
                    self.current_questionnaire = 'knee_injury'
                else:
                    self.current_questionnaire = 'knee_oa'
            elif 'shoulder' in last_user_message:
                # For shoulder, we don't have shoulder-specific questionnaires yet
                # Use a generic approach that doesn't rely on knee-specific scoring
                self.current_questionnaire = 'shoulder_generic'
            else:
                # Default to knee OA for now
                self.current_questionnaire = 'knee_oa'
        
        # Define state sequences based on questionnaire type
        # Comprehensive triage based on detailed questionnaire specifications
        if self.current_questionnaire == 'knee_injury':
            state_sequence = [
                TriageState.GREETING,
                TriageState.SELECT_BODY_PART,
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
                TriageState.GATHER_TREATMENT_RESPONSE,
                TriageState.GATHER_PREVIOUS_INJURY_SURGERY,
                TriageState.GATHER_LOCKING_TYPE,
                TriageState.GATHER_OVERUSE_CONTEXT,
                TriageState.GATHER_PHENOTYPE_SYMPTOMS,
                TriageState.GATHER_IMAGING_HISTORY,
                TriageState.GATHER_SMOKING_STATUS,
                TriageState.GATHER_SURGERY_INTEREST,
                TriageState.GATHER_CONSERVATIVE_TREATMENT_FAILURE,
                TriageState.GATHER_RED_FLAGS,
                TriageState.COMPLETE
            ]
        else:  # knee_oa
            state_sequence = [
                TriageState.GREETING,
                TriageState.SELECT_BODY_PART,
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
                TriageState.GATHER_OA_INDEX_DETAILED,
                TriageState.GATHER_FUNCTIONAL_IMPACT,
                TriageState.GATHER_PREVIOUS_TREATMENT,
                TriageState.GATHER_TREATMENT_RESPONSE,
                TriageState.GATHER_PREVIOUS_INJURY_SURGERY,
                TriageState.GATHER_LOCKING_TYPE,
                TriageState.GATHER_OVERUSE_CONTEXT,
                TriageState.GATHER_PHENOTYPE_SYMPTOMS,
                TriageState.GATHER_IMAGING_HISTORY,
                TriageState.GATHER_SMOKING_STATUS,
                TriageState.GATHER_SURGERY_INTEREST,
                TriageState.GATHER_CONSERVATIVE_TREATMENT_FAILURE,
                TriageState.GATHER_RED_FLAGS,
                TriageState.COMPLETE
            ]
        
        # Find the next question we need to ask based on what information we already have
        for state in state_sequence:
            # Skip if we've asked this question too many times (prevent infinite loops)
            if self.question_count.get(state, 0) >= 2:  # Reduced from 3 to 2
                continue
                
            if state == TriageState.GREETING:
                # Always start with greeting if no assistant messages yet (meaning we haven't greeted)
                if not any(msg['role'] == 'assistant' for msg in messages):
                    return state
            elif state == TriageState.SELECT_BODY_PART:
                # Ask for body part if we don't have it yet
                # Skip if we already have body part info from the patient's initial message
                if not self._has_body_part_info(patient_data, messages):
                    return state
            elif state == TriageState.GATHER_AGE:
                # Ask for age if we don't have it yet
                if not patient_data.get('patient', {}).get('age_years'):
                    return state
            elif state == TriageState.GATHER_LATERALITY:
                # Ask for laterality if we don't have it yet
                if not patient_data.get('laterality'):
                    return state
            elif state == TriageState.GATHER_DURATION:
                # Ask for duration if we don't have it yet
                if not patient_data.get('duration_class'):
                    return state
            elif state == TriageState.GATHER_MECHANISM:
                # Ask for mechanism if we don't have it yet
                if not patient_data.get('mechanism'):
                    return state
            elif state == TriageState.GATHER_SYMPTOMS:
                # Ask for symptoms if we don't have it yet
                if not patient_data.get('symptoms'):
                    return state
            elif state == TriageState.GATHER_PAIN_CHARACTER:
                # Ask for pain character if we don't have it yet
                if not patient_data.get('pain_character'):
                    return state
            elif state == TriageState.GATHER_RADIATION:
                # Ask for radiation if we don't have it yet
                if not patient_data.get('radiation'):
                    return state
            elif state == TriageState.GATHER_ASSOCIATED_SYMPTOMS:
                # Ask for associated symptoms if we don't have it yet
                if not patient_data.get('associated_symptoms'):
                    return state
            elif state == TriageState.GATHER_TIMING:
                # Ask for timing if we don't have it yet
                if not patient_data.get('timing'):
                    return state
            elif state == TriageState.GATHER_EXACERBATING_RELIEVING:
                # Ask for exacerbating/relieving factors if we don't have it yet
                if not patient_data.get('exacerbating_relieving'):
                    return state
            elif state == TriageState.GATHER_SEVERITY:
                # Ask for severity if we don't have it yet
                if not patient_data.get('severity'):
                    return state
            elif state == TriageState.GATHER_STIFFNESS:
                # Ask for stiffness if we don't have it yet (only for OA)
                if not patient_data.get('stiffness'):
                    return state
            elif state == TriageState.GATHER_KNEE_SCORE:
                # Ask for knee score if we don't have it yet (only for knee injury)
                if not patient_data.get('knee_score'):
                    return state
            elif state == TriageState.GATHER_FUNCTIONAL_IMPACT:
                # Ask for functional impact if we don't have it yet
                if not patient_data.get('functional_impact'):
                    return state
            elif state == TriageState.GATHER_PREVIOUS_TREATMENT:
                # Ask for previous treatment if we don't have it yet
                if not patient_data.get('previous_treatment'):
                    return state
            elif state == TriageState.GATHER_SURGERY_INTEREST:
                # Ask for surgery interest if we don't have it yet
                if not patient_data.get('surgery_interest'):
                    return state
            elif state == TriageState.GATHER_CONSERVATIVE_TREATMENT_FAILURE:
                # Ask for conservative treatment failure if we don't have it yet
                if not patient_data.get('conservative_treatment_failure'):
                    return state
            elif state == TriageState.GATHER_SMOKING_STATUS:
                # Ask for smoking status if we don't have it yet
                if not patient_data.get('smoking_status'):
                    return state
            elif state == TriageState.GATHER_PREVIOUS_INJURY_SURGERY:
                # Ask for previous injury/surgery if we don't have it yet
                if not patient_data.get('previous_injury_surgery'):
                    return state
            elif state == TriageState.GATHER_TREATMENT_RESPONSE:
                # Ask for treatment response if we don't have it yet
                if not patient_data.get('treatment_response'):
                    return state
            elif state == TriageState.GATHER_LOCKING_TYPE:
                # Ask for locking type if we don't have it yet
                if not patient_data.get('locking_type'):
                    return state
            elif state == TriageState.GATHER_OVERUSE_CONTEXT:
                # Ask for overuse context if we don't have it yet
                if not patient_data.get('overuse_context'):
                    return state
            elif state == TriageState.GATHER_OA_INDEX_DETAILED:
                # Ask for OA index detailed if we don't have it yet (only for OA)
                if not patient_data.get('oa_index_detailed'):
                    return state
            elif state == TriageState.GATHER_IMAGING_HISTORY:
                # Ask for imaging history if we don't have it yet
                if not patient_data.get('imaging_history'):
                    return state
            elif state == TriageState.GATHER_PHENOTYPE_SYMPTOMS:
                # Ask for phenotype symptoms if we don't have it yet
                if not patient_data.get('phenotype_symptoms'):
                    return state
            elif state == TriageState.GATHER_RED_FLAGS:
                # Ask for red flags if we don't have it yet
                if not patient_data.get('red_flags'):
                    return state
            elif state == TriageState.COMPLETE:
                # All information gathered
                return state
        
        return TriageState.COMPLETE

    def _has_body_part_info(self, patient_data: Dict[str, Any], messages: List[Dict]) -> bool:
        """Check if we have body part information from the conversation."""
        # Check if we have laterality (which indicates body part)
        if patient_data.get('laterality'):
            return True
        
        # Check if any message mentions body parts
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                if any(part in content for part in ['shoulder', 'knee', 'back', 'hip', 'ankle', 'wrist', 'elbow', 'neck', 'left', 'right']):
                    return True
        
        return False

    def _detect_hip_specific_patterns(self, patient_data: Dict[str, Any], messages: List[Dict]) -> Dict[str, Any]:
        """Detect hip-specific patterns for better clinical reasoning."""
        hip_patterns = {
            'groin_pain': False,
            'lateral_pain': False,
            'radiation_to_knee': False,
            'night_pain': False,
            'stiffness': False
        }
        
        # Check recent messages for hip-specific symptoms
        for message in messages[-3:]:  # Check last 3 messages
            if message['role'] == 'user':
                content = message.get('content', '').lower()
                
                # Groin pain patterns (hip OA, labral tear)
                if any(phrase in content for phrase in ['groin', 'inner thigh', 'pubic', 'inguinal']):
                    hip_patterns['groin_pain'] = True
                
                # Lateral pain patterns (greater trochanter bursitis)
                if any(phrase in content for phrase in ['side of hip', 'outer hip', 'greater trochanter', 'lateral hip']):
                    hip_patterns['lateral_pain'] = True
                
                # Radiation to knee (classic hip OA)
                if any(phrase in content for phrase in ['radiates to knee', 'pain down to knee', 'knee pain', 'thigh pain']):
                    hip_patterns['radiation_to_knee'] = True
                
                # Night pain (red flag for hip)
                if any(phrase in content for phrase in ['night pain', 'worse at night', 'can\'t sleep']):
                    hip_patterns['night_pain'] = True
                
                # Stiffness (hip OA pattern)
                if any(phrase in content for phrase in ['stiff', 'stiffness', 'hard to move']):
                    hip_patterns['stiffness'] = True
        
        return hip_patterns

    def _detect_spine_red_flags(self, patient_data: Dict[str, Any], messages: List[Dict]) -> Dict[str, Any]:
        """Detect spine-specific red flags for urgent referral."""
        red_flags = {
            'cancer_red_flags': False,
            'infection_red_flags': False,
            'cauda_equina': False,
            'fragility_fracture': False
        }
        
        # Check recent messages for red flags
        for message in messages[-3:]:  # Check last 3 messages
            if message['role'] == 'user':
                content = message.get('content', '').lower()
                age = patient_data.get('patient', {}).get('age_years', 0)
                
                # Cancer red flags: age >50, night pain, weight loss, past cancer
                if (age > 50 and 
                    any(phrase in content for phrase in ['night pain', 'worse at night', 'weight loss', 'lost weight', 'unexplained weight'])):
                    red_flags['cancer_red_flags'] = True
                
                # Infection red flags: fever, IV drug use, immunosuppression, hot tender spine
                if any(phrase in content for phrase in ['fever', 'hot', 'tender spine', 'immunosuppressed', 'diabetes']):
                    red_flags['infection_red_flags'] = True
                
                # Cauda equina: bladder/bowel/saddle anaesthesia
                if any(phrase in content for phrase in ['bladder', 'bowel', 'saddle', 'numbness', 'weakness', 'foot drop']):
                    red_flags['cauda_equina'] = True
                
                # Osteoporotic fracture: >65, sudden pain, female, fragility risk
                if (age > 65 and 
                    any(phrase in content for phrase in ['sudden', 'suddenly', 'acute', 'fragile', 'osteoporosis'])):
                    red_flags['fragility_fracture'] = True
        
        return red_flags

    def _extract_patient_data(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract structured patient data from conversation messages using simple keyword detection."""
        data = {
            "patient": {"age_years": None, "gender": None},
            "laterality": None,
            "duration_class": None,
            "mechanism": None,
            "symptoms": None,
            "pain_character": None,
            "radiation": None,
            "associated_symptoms": None,
            "timing": None,
            "exacerbating_relieving": None,
            "severity": None,
            "stiffness": None,
            "functional_impact": None,
            "previous_treatment": None,
            "red_flags": None,
            "detailed_treatment_history": None,
            "surgery_interest": None,
            "conservative_treatment_failure": None,
            "phenotype": [],
            # New comprehensive fields
            "smoking_status": None,
            "previous_injury_surgery": None,
            "treatment_response": None,
            "locking_type": None,
            "overuse_context": None,
            "oa_index_detailed": None,
            "imaging_history": None,
            "phenotype_symptoms": None
        }
        
        # Extract data from user messages
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content'].lower()
                
                # Extract age - improved pattern matching
                if any(word in content for word in ['age', 'years old', 'i am', 'i\'m', 'old']):
                    import re
                    # Look for patterns like "58 years old", "I'm 58", "age 58", etc.
                    age_patterns = [
                        r'(\d+)\s*years?\s*old',
                        r'i\'?m\s*(\d+)',
                        r'age\s*(\d+)',
                        r'(\d+)\s*year\s*old'
                    ]
                    for pattern in age_patterns:
                        age_match = re.search(pattern, content)
                        if age_match:
                            data["patient"]["age_years"] = int(age_match.group(1))
                            break
                
                # Extract gender - improved pattern matching with word boundaries
                import re
                if re.search(r'\b(female|woman|girl|she|her)\b', content):
                    data["patient"]["gender"] = "female"
                elif re.search(r'\b(male|man|boy|he|him)\b', content):
                    data["patient"]["gender"] = "male"
                
                # Extract laterality - improved pattern matching
                if 'left' in content:
                    data["laterality"] = "left"
                elif 'right' in content:
                    data["laterality"] = "right"
                elif any(word in content for word in ['middle', 'central', 'center', 'centered', 'both sides', 'both', 'bilateral']):
                    data["laterality"] = "bilateral"
                
                # Extract duration - improved pattern matching
                import re
                # Look for time patterns like "8 months", "2 weeks", "3 years", etc.
                time_patterns = [
                    r'(\d+)\s*months?',
                    r'(\d+)\s*weeks?',
                    r'(\d+)\s*years?'
                ]
                
                for pattern in time_patterns:
                    time_match = re.search(pattern, content)
                    if time_match:
                        value = int(time_match.group(1))
                        if 'month' in pattern:
                            if value < 3:  # Less than 3 months = subacute
                                data["duration_class"] = "subacute"
                            else:  # 3+ months = chronic
                                data["duration_class"] = "chronic"
                        elif 'week' in pattern:
                            if value < 2:  # Less than 2 weeks = acute
                                data["duration_class"] = "acute"
                            else:  # 2+ weeks = subacute
                                data["duration_class"] = "subacute"
                        elif 'year' in pattern:
                            data["duration_class"] = "chronic"
                        break
                
                # Fallback to keyword matching
                if not data["duration_class"]:
                    if any(word in content for word in ['acute', 'recent', 'just', 'today', 'yesterday']):
                        data["duration_class"] = "acute"
                    elif any(word in content for word in ['chronic', 'long time']):
                        data["duration_class"] = "chronic"
                    elif any(word in content for word in ['subacute']):
                        data["duration_class"] = "subacute"
                
                # Extract mechanism - improved detection
                mechanism_keywords = {
                    'twisting': ['injury', 'hurt', 'injured', 'accident', 'fall', 'twist', 'twisted', 'stepped off', 'landed', 'jumped', 'pivot', 'cutting', 'change of direction'],
                    'overuse': ['overuse', 'gradual', 'insidious', 'gradually', 'over time', 'slowly', 'training', 'running', 'exercise', 'repetitive'],
                    'direct_blow': ['blow', 'contact', 'collision', 'tackle', 'hit', 'struck', 'dashboard', 'fell onto'],
                    'unknown': ['sudden', 'suddenly', 'came on', 'woke up', 'not sure', 'don\'t know', 'unclear']
                }
                
                for mechanism, keywords in mechanism_keywords.items():
                    if any(word in content.lower() for word in keywords):
                        data["mechanism"] = mechanism
                        break
                
                # Extract symptoms
                if any(word in content for word in ['pain', 'ache', 'hurt', 'sore', 'discomfort', 'symptoms']):
                    data["symptoms"] = content
                
                # Extract pain character - improved pattern matching
                pain_keywords = [
                    'sharp', 'dull', 'aching', 'burning', 'throbbing', 'stabbing', 'stiff',
                    'crushing', 'pressure', 'intense', 'severe', 'excruciating', 'constant',
                    'constant pain', 'severe pain', 'intense pain', 'crushing pressure'
                ]
                if any(word in content for word in pain_keywords):
                    data["pain_character"] = content
                
                # Extract radiation - improved pattern matching
                radiation_keywords = [
                    'radiates', 'spreads', 'goes down', 'shoots', 'localized', 
                    'doesn\'t spread', 'no spread', 'just in', 'only in', 
                    'doesn\'t really spread', 'does not spread', 'travels', 
                    'down the lateral side', 'radiate down', 'spread to', 'goes to',
                    'doesn\'t really spread to', 'does not spread to'
                ]
                if any(word in content for word in radiation_keywords):
                    data["radiation"] = content
                
                # Extract associated symptoms
                if any(word in content for word in ['swelling', 'stiffness', 'numbness', 'weakness', 'clicking', 'popping', 'instability', 'locking']):
                    data["associated_symptoms"] = content
                
                # Extract timing - improved pattern matching
                if any(word in content for word in ['constant', 'comes and go', 'intermittent', 'episodic', 'consistent', 'getting better', 'gradually', 'improving', 'worse', 'better']):
                    data["timing"] = content
                
                # Extract exacerbating/relieving factors
                if any(word in content for word in ['better', 'worse', 'relief', 'rest', 'movement', 'activity', 'kneeling', 'bending', 'twisting']):
                    data["exacerbating_relieving"] = content
                
                # Extract severity - improved pattern matching
                import re
                # Look for pain scale patterns like "7/10", "8 out of 10", "rating 9", etc.
                severity_patterns = [
                    r'(\d+)\s*/\s*10',
                    r'(\d+)\s*out\s*of\s*10',
                    r'rating\s*(\d+)',
                    r'scale\s*(\d+)',
                    r'(\d+)\s*out\s*of\s*ten'
                ]
                
                for pattern in severity_patterns:
                    severity_match = re.search(pattern, content)
                    if severity_match:
                        # Extract the numeric value, not the whole sentence
                        data["severity"] = int(severity_match.group(1))
                        break
                
                # Fallback to keyword matching
                if not data["severity"]:
                    if any(word in content for word in ['scale', 'out of 10', 'rating', 'severity', '7 out of 10', '8 out of 10', '9 out of 10', '10 out of 10']):
                        data["severity"] = content
                
                # Extract stiffness
                if any(word in content for word in ['morning stiffness', 'stiff', 'loosen up']):
                    data["stiffness"] = content
                
                # Extract functional impact - improved pattern matching
                if any(word in content for word in ['work', 'daily activities', 'hobbies', 'difficulty', 'affecting', 'plumber', 'job', 'tasks', 'golf', 'playing', 'enjoy', 'frustrating', 'stuck', 'painful', 'swinging']):
                    data["functional_impact"] = content
                
                # Extract previous treatment - improved pattern matching
                treatment_keywords = [
                    'treatment', 'medication', 'physiotherapy', 'therapy', 'tried', 'analgesia', 
                    'knee support', 'stretching', 'exercises', 'foam rolling', 'ibuprofen', 
                    'paracetamol', 'pain relievers', 'over-the-counter', 'managing', 'self-managing'
                ]
                if any(word in content for word in treatment_keywords):
                    data["previous_treatment"] = content
                
                # Extract red flags
                if any(word in content for word in ['fever', 'chills', 'weight loss', 'unwell', 'hot joint']):
                    data["red_flags"] = content
                
                # Extract detailed treatment history
                if any(word in content for word in ['physiotherapy', 'physio', 'injection', 'steroid', 'specialist', 'specialist treatment', 'specialist treatments']):
                    data["detailed_treatment_history"] = content
                
                # Extract surgery interest - improved pattern matching
                surgery_keywords = [
                    'surgery', 'surgical', 'operation', 'yes', 'interested', 'consider', 
                    'recommended', 'if it\'s what I need', 'if it was recommended', 
                    'if that\'s what I need', 'if that was recommended', 'if necessary',
                    'if it\'s necessary', 'if that\'s necessary', 'if recommended'
                ]
                if any(word in content for word in surgery_keywords):
                    data["surgery_interest"] = content
                
                # Extract conservative treatment failure - improved pattern matching
                conservative_keywords = [
                    'tried', 'failed', 'didn\'t help', 'didn\'t work', 'no improvement', 
                    'helped', 'successful', 'effective', 'haven\'t tried', 'haven\'t had',
                    'no specialist treatments', 'no physiotherapy', 'no injections'
                ]
                if any(word in content for word in conservative_keywords):
                    data["conservative_treatment_failure"] = content
                
                # Extract symptoms/phenotype
                if 'instability' in content or 'giving way' in content:
                    data["phenotype"].append("instability")
                if 'locking' in content or 'catching' in content:
                    data["phenotype"].append("locking_catching")
                if 'anterior' in content or 'front' in content:
                    data["phenotype"].append("anterior_pain")
                
                # Extract smoking status
                if any(word in content for word in ['smoke', 'smoking', 'cigarette', 'tobacco', 'non-smoker', 'never smoked']):
                    data["smoking_status"] = content
                
                # Extract previous injury/surgery
                if any(word in content for word in ['acl', 'meniscus', 'arthroscopy', 'knee replacement', 'surgery', 'operation', 'reconstruction']):
                    data["previous_injury_surgery"] = content
                elif any(phrase in content for phrase in ['no previous', 'no injuries', 'no surgeries', 'haven\'t had', 'no operations']):
                    data["previous_injury_surgery"] = "none"
                
                # Extract treatment response
                if any(word in content for word in ['helped', 'better', 'improved', 'no change', 'worse', 'didn\'t help', 'no difference']):
                    data["treatment_response"] = content
                
                # Extract locking type
                if any(phrase in content for phrase in ['stuck', 'won\'t move', 'locked', 'completely stuck']):
                    data["locking_type"] = "true_lock"
                elif any(phrase in content for phrase in ['click', 'catch', 'brief', 'pops', 'snaps']):
                    data["locking_type"] = "catch_click"
                
                # Extract overuse context
                if any(phrase in content for phrase in ['running', 'marathon', 'mileage', 'training', 'hill repeats', 'prolonged standing']):
                    data["overuse_context"] = "running_overuse"
                
                # Extract OA index detailed
                if any(word in content for word in ['stairs', 'chair', 'car', 'socks', 'bath', 'domestic', 'bending']):
                    data["oa_index_detailed"] = content
                
                # Extract imaging history
                if any(word in content for word in ['x-ray', 'mri', 'scan', 'imaging', 'radiograph']):
                    data["imaging_history"] = content
                
                # Extract phenotype symptoms
                if any(phrase in content for phrase in ['instability', 'giving way', 'locking', 'catching', 'front of knee', 'behind kneecap']):
                    data["phenotype_symptoms"] = content
        
        return data

    async def get_next_response(self, messages: List[Dict]) -> str:
        """
        Deterministically emit the next question from the state machine.
        No LLM is used for asking questions. This prevents persona drift/hallucinations.
        """
        current_state = self._determine_current_state(messages)

        # Track question count to prevent infinite loops
        self.question_count[current_state] = self.question_count.get(current_state, 0) + 1

        # Conversation complete
        if current_state == TriageState.COMPLETE:
            return ("Thank you for sharing all the information with me. "
                    "I now have a complete picture of your situation. "
                    "A clinical summary with differential diagnosis will be prepared for the clinical team at SWLEOC "
                    "to direct you to the most appropriate care pathway.")

        # Get the exact question/task text
        task_prompt = self._get_prompt_for_state(current_state)

        # If the prompt includes 'Say exactly:' return that literal content
        if "Say exactly:" in task_prompt:
            # Extract the quoted string after "Say exactly:"
            start = task_prompt.find("Say exactly:") + len("Say exactly:")
            text = task_prompt[start:].strip()
            # Strip surrounding quotes if present
            if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
                text = text[1:-1]
            return text

        # Otherwise, return the task prompt as-is (you can add a tiny friendly lead-in if you like)
        return task_prompt

# Methodology: AI-Powered Musculoskeletal Triage System for SWLEOC

## 1. System Overview

This research presents the development and implementation of an AI-powered musculoskeletal (MSK) triage system designed for the Southwest London Elective Orthopaedic Centre (SWLEOC). The system employs a conversational AI agent to conduct structured patient assessments and provide clinical triage recommendations.

## 2. System Architecture

### 2.1 Microservices Architecture

The system is built using a microservices architecture with three primary components:

1. **Triage Agent Service** (`triage_agent.py`): Manages the conversational flow and state machine
2. **Summarization Agent Service** (`summarization_agent.py`): Generates clinical summaries and triage recommendations
3. **User Interface Service** (`ui.py`): Provides the web-based chat interface

### 2.2 Technology Stack

- **Backend Framework**: FastAPI (Python 3.11)
- **Frontend Interface**: Streamlit
- **Language Model**: Ollama with Llama 3.1 8B model
- **Containerization**: Docker with Docker Compose
- **Communication**: HTTP REST APIs between services
- **GPU Acceleration**: NVIDIA GPU support for LLM inference

### 2.3 Infrastructure Design

The system is deployed using Docker containers with the following configuration:
- **triage_app**: FastAPI backend service (Port 8000)
- **llm_server**: Ollama LLM server (Port 11434)
- **triage_ui**: Streamlit frontend (Port 8501)

## 3. Clinical Assessment Framework

### 3.1 SOCRATES Methodology Integration

The system implements the standardized SOCRATES pain assessment framework:

- **S**ite: Location of the problem
- **O**nset: When and how the problem started
- **C**haracter: Description of the symptom
- **R**adiation: Whether the feeling spreads to other areas
- **A**ssociations: Other symptoms that occur simultaneously
- **T**iming: Pattern and frequency of symptoms
- **E**xacerbating/Relieving factors: What makes it better or worse
- **S**everity: Pain rating on a 0-10 scale

### 3.2 Triage-Specific Assessment

Beyond SOCRATES, the system gathers additional triage-relevant information:

1. **Injury Mechanism**: How the problem started
2. **Red Flags**: Screening for serious conditions (fever, weight loss, neurological symptoms)
3. **Previous Treatment**: What interventions have been tried
4. **Functional Impact**: Effect on daily activities, work, and sleep

## 4. State Machine Implementation

### 4.1 Conversational Flow Control

The system uses a finite state machine to ensure systematic data collection:

```python
class TriageState(str, Enum):
    GREETING = "GREETING"
    GATHER_SITE = "GATHER_SITE"
    GATHER_ONSET = "GATHER_ONSET"
    GATHER_CHARACTER = "GATHER_CHARACTER"
    GATHER_RADIATION = "GATHER_RADIATION"
    GATHER_ASSOCIATIONS = "GATHER_ASSOCIATIONS"
    GATHER_TIMING = "GATHER_TIMING"
    GATHER_EXACERBATING_RELIEVING = "GATHER_EXACERBATING_RELIEVING"
    GATHER_SEVERITY = "GATHER_SEVERITY"
    GATHER_INJURY_MECHANISM = "GATHER_INJURY_MECHANISM"
    GATHER_RED_FLAGS = "GATHER_RED_FLAGS"
    GATHER_PREVIOUS_TREATMENT = "GATHER_PREVIOUS_TREATMENT"
    GATHER_FUNCTIONAL_IMPACT = "GATHER_FUNCTIONAL_IMPACT"
    COMPLETE = "COMPLETE"
```

### 4.2 State Transition Logic

The system determines the current state based on the number of user responses, ensuring a sequential progression through all assessment domains.

## 5. AI Agent Design

### 5.1 Triage Agent

The TriageAgent class manages the conversational interface with the following key features:

- **Context-Aware Prompting**: Each state has a specific prompt template
- **Empathetic Communication**: Maintains professional and understanding tone
- **Single Question Focus**: Asks one question at a time to avoid confusion
- **Error Handling**: Graceful degradation when LLM services are unavailable

### 5.2 Summarization Agent

The SummarizationAgent processes completed conversations to generate:

1. **Structured Clinical Summary**: Organized using SOCRATES framework
2. **Triage Recommendation**: One of four pathways:
   - Urgent Care/A&E (fractures, dislocations, severe neurological symptoms)
   - GP/Primary Care (red flags, unclear diagnosis)
   - MSK Physiotherapy (mechanical pain, strains, chronic issues)
   - SWLEOC Orthopaedic Surgery (failed conservative treatment)

3. **Clinical Justification**: Reasoning for the triage decision

## 6. Data Flow and Processing

### 6.1 Conversation Management

1. **User Input**: Captured through Streamlit chat interface
2. **State Determination**: System identifies current assessment stage
3. **Prompt Generation**: Contextual prompts created for LLM
4. **Response Generation**: Ollama LLM generates appropriate questions
5. **State Progression**: System advances to next assessment domain

### 6.2 Summary Generation Process

1. **Conversation Analysis**: Full transcript processed by SummarizationAgent
2. **Clinical Extraction**: Key information extracted and structured
3. **Triage Decision**: AI determines appropriate care pathway
4. **Report Generation**: Structured clinical summary with recommendations

## 7. Quality Assurance and Safety Measures

### 7.1 Clinical Safety

- **Red Flag Screening**: Systematic assessment for serious conditions
- **No Diagnosis**: System explicitly states it provides information gathering only
- **Professional Oversight**: All recommendations require clinical review
- **Conservative Approach**: When in doubt, directs to higher levels of care

### 7.2 Technical Reliability

- **Error Handling**: Graceful degradation when services fail
- **Timeout Management**: Prevents system hanging on LLM requests
- **State Persistence**: Maintains conversation context throughout session
- **Input Validation**: Ensures proper data format and structure

## 8. Evaluation Methodology

### 8.1 Synthetic Data Generation from Retrospective Clinic Letters

#### 8.1.1 Data Source and Processing
The evaluation methodology employs retrospective patient clinic letters as the foundation for generating synthetic conversations. This approach provides several advantages:

- **Rich Clinical Data**: Clinic letters contain structured information that maps directly to the SOCRATES framework
- **Ground Truth Availability**: Actual clinical outcomes and triage decisions are known
- **Large Dataset Potential**: Hundreds of test cases can be generated from existing records
- **Ethical Compliance**: No patient risk during system testing and development

#### 8.1.2 Data Extraction Pipeline
The clinic letter processing follows a systematic approach:

1. **Structured Data Extraction**: Information is extracted into standardized format:
   ```python
   clinic_letter_data = {
       "presenting_complaint": "Primary symptom description",
       "socrates": {
           "site": "Anatomical location",
           "onset": "Temporal characteristics",
           "character": "Symptom description",
           "radiation": "Spread patterns",
           "associations": "Concurrent symptoms",
           "timing": "Frequency and patterns",
           "exacerbating": "Worsening factors",
           "relieving": "Improving factors",
           "severity": "Pain scale rating"
       },
       "triage_info": {
           "injury_mechanism": "Causative factors",
           "red_flags": "Serious condition indicators",
           "previous_treatment": "Prior interventions",
           "functional_impact": "Activity limitations"
       },
       "clinical_outcome": "Actual triage pathway",
       "final_diagnosis": "Clinical diagnosis"
   }
   ```

2. **Anonymization Process**: 
   - Removal of all personally identifiable information (PII)
   - Replacement of specific dates with relative timeframes
   - Generalization of locations while preserving clinical relevance
   - Maintenance of clinical accuracy while ensuring privacy compliance

#### 8.1.3 Synthetic Conversation Generation
The synthetic conversation generation employs a multi-step process:

1. **Template-Based Generation**: Conversations follow the established state machine progression
2. **Natural Language Variation**: Multiple conversation styles generated from identical clinical data
3. **Realistic Patient Responses**: Simulation of how patients might describe symptoms in natural language
4. **Clinical Accuracy Validation**: Expert review ensures medical accuracy and realism

#### 8.1.4 Quality Assurance
- **Clinical Expert Review**: Validation of conversation realism and medical accuracy
- **Information Completeness**: Verification that all SOCRATES elements are captured
- **Triage Pathway Validation**: Confirmation that generated recommendations match clinical outcomes

### 8.2 Clinical Validation

The system's triage recommendations are evaluated against:
- **Synthetic Conversation Outcomes**: Comparison with known clinical pathways from source letters
- **Expert Clinician Assessments**: Independent review of AI-generated summaries
- **Triage Pathway Accuracy**: Correct assignment to appropriate care levels
- **Information Extraction Completeness**: Verification of comprehensive data capture

### 8.3 Technical Performance

Key metrics include:
- **Response Time and System Availability**: System reliability and performance
- **Conversation Completion Rates**: Successful progression through all assessment states
- **Clinical Summary Accuracy**: Quality of generated clinical summaries
- **Triage Recommendation Precision**: Accuracy of care pathway assignments

## 9. Ethical Considerations

### 9.1 Patient Privacy

- No persistent storage of patient data
- Session-based data handling
- Compliance with healthcare data protection regulations

### 9.2 Clinical Responsibility

- Clear communication of AI limitations
- Emphasis on information gathering vs. diagnosis
- Requirement for human clinical oversight
- Transparent triage reasoning

## 10. Implementation Considerations

### 10.1 Scalability

- Microservices architecture supports horizontal scaling
- GPU-accelerated LLM inference for performance
- Containerized deployment for easy scaling

### 10.2 Integration

- RESTful API design enables integration with existing healthcare systems
- Standardized data formats for interoperability
- Modular design allows component replacement or enhancement

## 11. Future Enhancements

### 11.1 Clinical Improvements

- Integration with electronic health records
- Multi-language support for diverse patient populations
- Specialized assessment modules for different MSK conditions
- Integration with imaging and diagnostic data

### 11.2 Technical Advancements

- Fine-tuned models for medical domain expertise
- Real-time clinical decision support
- Integration with telemedicine platforms
- Advanced natural language processing for symptom recognition

This methodology provides a comprehensive framework for understanding the development, implementation, and evaluation of the AI-powered MSK triage system, ensuring both clinical effectiveness and technical robustness.

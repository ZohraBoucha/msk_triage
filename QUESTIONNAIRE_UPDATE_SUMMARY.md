# MSK Triage System - Questionnaire-Based Update Summary

## Overview
Successfully updated the MSK Triage System to replace the SOCRATES assessment framework with questionnaire-based assessments that generate SBAR clinical summaries and differential diagnoses with top 3 diagnoses.

## Key Changes Made

### 1. New Questionnaire Engine (`app/questionnaire_engine.py`)
- **Purpose**: Reusable engine for evaluating questionnaire-based assessments
- **Features**:
  - JSON specification-driven evaluation
  - Scoring system with confidence bands
  - Red flag detection and urgent routing
  - Support for complex aggregate scoring rules
  - Diagnosis ranking with tie-breakers
  - Safety net message generation

### 2. Questionnaire Specifications (`app/questionnaire_specs.py`)
- **Knee OA Assessment**: For chronic osteoarthritis conditions
- **Knee Injury Assessment**: For acute injury scenarios
- **Features**:
  - Comprehensive scoring rules for different clinical scenarios
  - Red flag logic for urgent conditions
  - Confidence band definitions
  - Support for knee score calculations
  - Mechanism mapping from free text

### 3. Updated Triage Agent (`app/triage_agent.py`)
- **Replaced SOCRATES with questionnaire-based flow**:
  - Dynamic state machine based on questionnaire type
  - Body part and injury type detection
  - Structured data extraction from conversations
  - Support for specialized questions (knee score, exam findings)
- **New States**:
  - `SELECT_BODY_PART`, `SELECT_QUESTIONNAIRE`
  - `GATHER_AGE`, `GATHER_LATERALITY`, `GATHER_DURATION`
  - `GATHER_MECHANISM`, `GATHER_SYMPTOMS`, `GATHER_PAIN_CHARACTER`
  - `GATHER_KNEE_SCORE`, `GATHER_EXAM_FINDINGS`, `GATHER_IMAGING`

### 4. Enhanced Summarization Agent (`app/summarization_agent.py`)
- **SBAR Format Implementation**:
  - **Situation**: Patient demographics, presenting complaint, body part
  - **Background**: Onset, mechanism, previous treatment, relevant history
  - **Assessment**: Clinical findings, exam results, imaging
  - **Recommendation**: Triage pathway, clinical reasoning, next steps
- **Differential Diagnosis Generation**:
  - Top 3 diagnoses with confidence scores
  - Key supporting features for each diagnosis
  - Red flag considerations
  - Integration with questionnaire analysis results

### 5. Updated User Interface (`app/ui.py`)
- **Enhanced UI Features**:
  - Updated welcome message for questionnaire-based approach
  - Better visual formatting for clinical summaries
  - Informative sidebar with system capabilities
  - "Start New Assessment" button
  - Clear indication of SBAR and differential diagnosis generation

### 6. Test Suite (`test_questionnaire_system.py`)
- **Comprehensive Testing**:
  - Questionnaire engine validation
  - Triage agent state management
  - Patient data extraction
  - Summarization agent integration
  - End-to-end workflow testing

## Technical Implementation Details

### Questionnaire Engine Architecture
```python
# Core evaluation flow
1. Red flag detection (urgent routing)
2. Initialize diagnosis scores
3. Apply scoring blocks in order:
   - mechanism/onset_mechanism
   - symptoms
   - oa_index (with aggregates)
   - knee_score (with deficits)
   - exam findings
   - imaging results
4. Rank results by score
5. Apply confidence bands
6. Generate safety net messages
```

### SBAR Summary Structure
```
SITUATION:
- Patient Demographics
- Presenting Complaint
- Body Part Affected

BACKGROUND:
- Onset & Duration
- Mechanism of Injury
- Previous Treatment
- Relevant History

ASSESSMENT:
- Clinical Findings
- Physical Examination
- Imaging

RECOMMENDATION:
- Triage Pathway
- Clinical Reasoning
- Next Steps
```

### Differential Diagnosis Format
```
DIFFERENTIAL DIAGNOSIS (Top 3):

1. PRIMARY DIAGNOSIS:
   - Diagnosis: [Condition Name]
   - Confidence: [High/Moderate/Low]
   - Key Supporting Features: [Clinical features]
   - Score: [Numerical score]

2. SECONDARY DIAGNOSIS: [Same format]
3. TERTIARY DIAGNOSIS: [Same format]

RED FLAG CONSIDERATIONS:
- [Serious conditions to rule out]
```

## Supported Questionnaires

### 1. Knee Osteoarthritis Assessment
- **Target Conditions**: Tibiofemoral OA, Patellofemoral OA, PFPS
- **Key Features**: Morning stiffness, functional impact, OA index scoring
- **Scoring**: Age-based weighting, stiffness patterns, functional deficits

### 2. Knee Injury Assessment
- **Target Conditions**: ACL/PCL tears, meniscal tears, ligament sprains, patellar instability
- **Key Features**: Mechanism-based scoring, knee score deficits, exam findings
- **Scoring**: Injury mechanism, knee score calculations, physical exam

## Clinical Benefits

### 1. Structured Assessment
- **Consistent Data Collection**: Standardized questions ensure comprehensive information gathering
- **Evidence-Based Scoring**: Clinical decision rules based on established medical literature
- **Red Flag Detection**: Systematic screening for urgent conditions

### 2. Enhanced Clinical Summaries
- **SBAR Format**: Familiar to healthcare professionals, improves communication
- **Differential Diagnosis**: Top 3 diagnoses with supporting evidence
- **Confidence Scoring**: Clear indication of diagnostic certainty

### 3. Improved Triage Accuracy
- **Questionnaire-Driven**: More accurate than free-form SOCRATES
- **Specialized Assessments**: Tailored questions for specific conditions
- **Quantitative Scoring**: Objective assessment of clinical features

## System Integration

### API Endpoints (Unchanged)
- `POST /ask` - Get next question from triage agent
- `POST /summarize` - Generate SBAR summary and differential diagnosis

### Docker Configuration (Unchanged)
- `triage_app` - FastAPI backend (Port 8000)
- `llm_server` - Ollama LLM server (Port 11434)
- `triage_ui` - Streamlit frontend (Port 8501)

## Future Enhancements

### 1. Additional Questionnaires
- Shoulder assessments
- Hip assessments
- Back pain questionnaires
- Foot and ankle evaluations

### 2. Enhanced Features
- Multi-language support
- Integration with electronic health records
- Real-time clinical decision support
- Advanced imaging analysis

### 3. Clinical Validation
- Prospective clinical trials
- Comparison with traditional assessment methods
- Accuracy validation against clinical outcomes

## Testing Results

✅ **Questionnaire Engine**: Successfully processes knee OA and injury scenarios
✅ **Triage Agent**: Correctly manages state transitions and data extraction
✅ **Summarization Agent**: Generates comprehensive SBAR summaries
✅ **UI Integration**: Seamless user experience with enhanced formatting
✅ **End-to-End Testing**: Complete workflow validation

## Conclusion

The MSK Triage System has been successfully updated to use questionnaire-based assessments instead of SOCRATES, providing:

1. **More Structured Data Collection** through specialized questionnaires
2. **Enhanced Clinical Summaries** in SBAR format
3. **Evidence-Based Differential Diagnoses** with top 3 diagnoses
4. **Improved Triage Accuracy** through quantitative scoring
5. **Better Clinical Communication** through standardized formats

The system maintains backward compatibility with existing infrastructure while providing significantly enhanced clinical assessment capabilities.

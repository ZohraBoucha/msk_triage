"""
Questionnaire Specifications for MSK Triage System

This module contains JSON specifications for different questionnaire forms
used in the MSK triage system.
"""

# Knee OA/Injury Triage Decision Engine Specification
KNEE_OA_SPEC = {
    "version": "1.0",
    "name": "Knee OA / Injury Triage Decision Engine",
    "diagnoses": [
        "tibiofemoral_oa", "patellofemoral_oa", "pfps", "acl_tear", "pcl_tear",
        "medial_meniscal_tear", "lateral_meniscal_tear", "mcl_sprain", "lcl_sprain",
        "patellar_instability", "painful_arthroplasty", "bakers_cyst", "loose_body"
    ],
    "ranking": {
        "top_k": 3,
        "tie_breakers": [
            {"rule": "prefer_exam_positive_over_symptom_only"},
            {"rule": "prefer_mechanism_consistent"},
            {"rule": "prefer_side_specific_over_general"}
        ],
        "justification": {"max_reasons_per_dx": 4, "reason_selection": "highest_weighted_features"}
    },
    "output": {
        "fields": ["diagnosis_code", "score", "confidence_band", "key_drivers", "safety_net"],
        "confidence_bands": [
            {"min": 0, "max": 7, "label": "low"},
            {"min": 8, "max": 15, "label": "moderate"},
            {"min": 16, "max": 1000, "label": "high"}
        ],
        "safety_net_rules": [
            {"if": "any_red_flag_triggered", "message": "Urgent same-day assessment recommended."},
            {"if": "diagnosis_includes:painful_arthroplasty", "message": "Consider infection screen (CRP/ESR), targeted imaging, and arthroplasty review."}
        ]
    },
    "red_flag_logic": [
        {"if_all_true": ["red_flags.fever_unwell_hot_joint"], "action": {"route": "urgent", "diagnosis": "septic_arthritis", "override_ranking": True}},
        {"if_all_true": ["red_flags.true_locked_knee"], "action": {"route": "urgent", "diagnosis": "bucket_handle_meniscal_tear", "override_ranking": True}},
        {"if_all_true": ["red_flags.inability_slr_after_eccentric_load"], "action": {"route": "urgent", "diagnosis": "extensor_mechanism_rupture", "override_ranking": True}}
    ],
    "scoring": {
        "onset_mechanism": [
            {"when": {"duration_class": "acute", "mechanism": ["twisting", "pivot"]}, "add": {"acl_tear": 3, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2, "pcl_tear": 1}},
            {"when": {"duration_class": "acute", "mechanism": ["direct_blow"], "exam.alignment": "valgus"}, "add": {"mcl_sprain": 3, "medial_meniscal_tear": 1}},
            {"when": {"duration_class": "acute", "mechanism": ["direct_blow"], "exam.alignment": "varus"}, "add": {"lcl_sprain": 3, "lateral_meniscal_tear": 1}},
            {"when": {"mechanism": ["overuse"]}, "add": {"tibiofemoral_oa": 2, "patellofemoral_oa": 2, "pfps": 1}},
            {"when": {"patient.age_years": ">=45", "duration_class": "chronic"}, "add": {"tibiofemoral_oa": 3, "patellofemoral_oa": 2}},
            {"when": {"mechanism": ["post_op"]}, "add": {"painful_arthroplasty": 4}}
        ],
        "symptoms": [
            {"when": {"phenotype": "instability"}, "add": {"acl_tear": 3, "patellar_instability": 2, "pcl_tear": 1}},
            {"when": {"phenotype": "locking_catching"}, "add": {"medial_meniscal_tear": 3, "lateral_meniscal_tear": 3, "loose_body": 2, "tibiofemoral_oa": 1}},
            {"when": {"phenotype": "anterior_pain"}, "add": {"pfps": 3, "patellofemoral_oa": 2, "tibiofemoral_oa": 1}},
            {"when": {"oa_pattern": "morning_stiffness_<30min"}, "add": {"tibiofemoral_oa": 2, "patellofemoral_oa": 1}}
        ],
        "oa_index": [
            {"when": {"oa_index.global_pain": "mild"}, "add_all": 1},
            {"when": {"oa_index.global_pain": "moderate"}, "add_all": 2},
            {"when": {"oa_index.global_pain": "severe"}, "add_all": 3},
            {"when": {"oa_index.global_pain": "unbearable"}, "add_all": 4},
            {"when": {"oa_index.stiffness_morning": ["moderate", "severe", "unbearable"]}, "add": {"tibiofemoral_oa": 2, "patellofemoral_oa": 1}},
            {"when": {"oa_index.stiffness_after_rest": ["moderate", "severe", "unbearable"]}, "add": {"tibiofemoral_oa": 2, "patellofemoral_oa": 1}},
            {
                "when": {"oa_index.function": "any"},
                "aggregate": [
                    {"method": "sum_function_items", "map": {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "unbearable": 3}, "then_add": {"tibiofemoral_oa": "floor(total/6)"}},
                    {"method": "sum_pf_loaded_items", "items": ["stairs_down", "stairs_up", "rise_from_sit", "in_out_car", "socks_on_off"], "map": {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "unbearable": 3}, "then_add": {"patellofemoral_oa": "floor(total/4)", "pfps": "floor(total/5)"}}
                ]
            }
        ],
        "exam": [
            {"when": {"exam.lachman": "yes_soft_endpoint"}, "add": {"acl_tear": 6}},
            {"when": {"exam.lachman": "yes_firm_endpoint"}, "add": {"acl_tear": 3}},
            {"when": {"exam.pivot_shift": True}, "add": {"acl_tear": 4}},
            {"when": {"exam.posterior_drawer": True}, "add": {"pcl_tear": 6}},
            {"when": {"exam.mcmurray_medial": True}, "add": {"medial_meniscal_tear": 4}},
            {"when": {"exam.mcmurray_lateral": True}, "add": {"lateral_meniscal_tear": 4}},
            {"when": {"exam.joint_line_tenderness_medial": True}, "add": {"medial_meniscal_tear": 2}},
            {"when": {"exam.joint_line_tenderness_lateral": True}, "add": {"lateral_meniscal_tear": 2}},
            {"when": {"exam.mcl_laxity_grade": "grade1"}, "add": {"mcl_sprain": 2}},
            {"when": {"exam.mcl_laxity_grade": "grade2"}, "add": {"mcl_sprain": 3}},
            {"when": {"exam.mcl_laxity_grade": "grade3"}, "add": {"mcl_sprain": 4}},
            {"when": {"exam.lcl_laxity_grade": "grade1"}, "add": {"lcl_sprain": 2}},
            {"when": {"exam.lcl_laxity_grade": "grade2"}, "add": {"lcl_sprain": 3}},
            {"when": {"exam.lcl_laxity_grade": "grade3"}, "add": {"lcl_sprain": 4}},
            {"when": {"exam.pf_compression_clarke": True}, "add": {"patellofemoral_oa": 3, "pfps": 2}},
            {"when": {"exam.patellar_apprehension": True}, "add": {"patellar_instability": 4}},
            {"when": {"exam.j_sign": True}, "add": {"patellar_instability": 3}},
            {"when": {"exam.patella_alta": True}, "add": {"patellar_instability": 2}},
            {"when": {"exam.pf_crepitus": "chondral"}, "add": {"patellofemoral_oa": 2, "pfps": 2}},
            {"when": {"exam.pf_crepitus": "bone"}, "add": {"patellofemoral_oa": 3}},
            {"when": {"exam.alignment": "varus"}, "add": {"tibiofemoral_oa": 3}},
            {"when": {"exam.alignment": "valgus"}, "add": {"tibiofemoral_oa": 3}},
            {"when": {"exam.effusion": "mild", "duration_class": "acute"}, "add": {"acl_tear": 1, "pcl_tear": 1, "medial_meniscal_tear": 1, "lateral_meniscal_tear": 1}},
            {"when": {"exam.effusion": "moderate", "duration_class": "acute"}, "add": {"acl_tear": 2, "pcl_tear": 2, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2}},
            {"when": {"exam.effusion": "severe", "duration_class": "acute"}, "add": {"acl_tear": 3, "pcl_tear": 2, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2}},
            {"when": {"exam.effusion": "mild", "duration_class": ["subacute", "chronic"]}, "add": {"tibiofemoral_oa": 1}},
            {"when": {"exam.effusion": "moderate", "duration_class": ["subacute", "chronic"]}, "add": {"tibiofemoral_oa": 1}},
            {"when": {"exam.effusion": "severe", "duration_class": ["subacute", "chronic"]}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"exam.rom_restriction": True}, "add": {"tibiofemoral_oa": 2, "medial_meniscal_tear": 1, "lateral_meniscal_tear": 1}},
            {"when": {"exam.fixed_flexion_deformity": True}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"exam.quads_tone": "reduced"}, "add": {"tibiofemoral_oa": 1, "pfps": 1}},
            {"when": {"exam.bakers_pseudocyst": True}, "add": {"bakers_cyst": 4}}
        ],
        "imaging": [
            {"when": {"imaging.xray_oa_tf": True}, "add": {"tibiofemoral_oa": 6}},
            {"when": {"imaging.xray_oa_pf": True}, "add": {"patellofemoral_oa": 4}},
            {"when": {"imaging.malalignment": "varus"}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"imaging.malalignment": "valgus"}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"imaging.mri_acl": True}, "add": {"acl_tear": 8}},
            {"when": {"imaging.mri_pcl": True}, "add": {"pcl_tear": 8}},
            {"when": {"imaging.mri_medial_meniscus": True}, "add": {"medial_meniscal_tear": 8}},
            {"when": {"imaging.mri_lateral_meniscus": True}, "add": {"lateral_meniscal_tear": 8}}
        ]
    }
}

# Knee Injury Assessment Sheet Specification
KNEE_INJURY_SPEC = {
    "version": "1.0",
    "name": "Knee Injury Triage Decision Engine",
    "source_form": "KNEE INJURY ASSESSMENT SHEET",
    "diagnoses": [
        "acl_tear", "pcl_tear", "mcl_sprain", "lcl_sprain",
        "medial_meniscal_tear", "lateral_meniscal_tear",
        "patellar_instability", "pfps", "patellofemoral_oa",
        "tibiofemoral_oa", "bakers_cyst", "loose_body", "painful_arthroplasty"
    ],
    "red_flag_logic": [
        {"if_all_true": ["red_flags.fever_unwell_hot_joint"], "action": {"route": "urgent", "diagnosis": "septic_arthritis", "override_ranking": True}},
        {"if_all_true": ["red_flags.true_locked_knee"], "action": {"route": "urgent", "diagnosis": "bucket_handle_meniscal_tear", "override_ranking": True}},
        {"if_all_true": ["red_flags.inability_slr_after_eccentric_load"], "action": {"route": "urgent", "diagnosis": "extensor_mechanism_rupture", "override_ranking": True}}
    ],
    "scoring": {
        "mechanism": [
            {"when": {"mechanism": ["twisting", "pivot"]}, "add": {"acl_tear": 3, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2, "pcl_tear": 1}},
            {"when": {"mechanism": ["direct_blow"], "exam.alignment": "valgus"}, "add": {"mcl_sprain": 3, "medial_meniscal_tear": 1}},
            {"when": {"mechanism": ["direct_blow"], "exam.alignment": "varus"}, "add": {"lcl_sprain": 3, "lateral_meniscal_tear": 1}},
            {"when": {"mechanism": ["non_contact_jump_land"]}, "add": {"acl_tear": 2, "patellar_instability": 1}},
            {"when": {"mechanism": ["overuse"]}, "add": {"pfps": 2, "patellofemoral_oa": 1, "tibiofemoral_oa": 1}},
            {"when": {"mechanism": ["post_op"]}, "add": {"painful_arthroplasty": 4}}
        ],
        "knee_score": [
            {"when": {"knee_score": "present"}, "aggregate": [
                {"method": "deficit", "field": "instability", "max": 25, "scale": 0.2, "then_add": {"acl_tear": "round(scale*deficit)", "patellar_instability": "round(0.8*scale*deficit)", "pcl_tear": "round(0.3*scale*deficit)"}},
                {"method": "deficit", "field": "locking", "max": 15, "scale": 0.3, "then_add": {"medial_meniscal_tear": "round(scale*deficit)", "lateral_meniscal_tear": "round(0.9*scale*deficit)", "loose_body": "round(0.6*scale*deficit)"}},
                {"method": "deficit", "field": "swelling", "max": 10, "scale": 0.2, "then_add": {"acl_tear": "round(scale*deficit)", "pcl_tear": "round(0.5*scale*deficit)", "medial_meniscal_tear": "round(0.5*scale*deficit)", "lateral_meniscal_tear": "round(0.5*scale*deficit)", "tibiofemoral_oa": "round(0.3*scale*deficit)"}},
                {"method": "deficit", "field": "stair_climbing", "max": 10, "scale": 0.2, "then_add": {"pfps": "round(scale*deficit)", "patellofemoral_oa": "round(0.8*scale*deficit)"}},
                {"method": "deficit", "field": "squatting", "max": 5, "scale": 0.3, "then_add": {"medial_meniscal_tear": "round(scale*deficit)", "lateral_meniscal_tear": "round(0.8*scale*deficit)", "pfps": "round(0.6*scale*deficit)"}},
                {"method": "deficit", "field": "support", "max": 5, "scale": 0.4, "then_add": {"acl_tear": "round(scale*deficit)", "pcl_tear": "round(0.5*scale*deficit)", "mcl_sprain": "round(0.6*scale*deficit)", "lcl_sprain": "round(0.6*scale*deficit)"}},
                {"method": "deficit", "field": "limp", "max": 5, "scale": 0.2, "then_add": {"tibiofemoral_oa": "round(scale*deficit)", "pfps": "round(0.6*scale*deficit)"}},
                {"method": "total", "scale": 0.1, "then_add": {"tibiofemoral_oa": "round(scale*(100-total))"}}
            ]}
        ],
        "symptoms_from_text": [
            {"when": {"impact_on_activities_text": "mentions_instability"}, "add": {"acl_tear": 2, "patellar_instability": 2}},
            {"when": {"injury_mechanism_text": "mentions_locking"}, "add": {"medial_meniscal_tear": 2, "lateral_meniscal_tear": 2}}
        ],
        "exam": [
            {"when": {"exam.lachman": "yes_soft_endpoint"}, "add": {"acl_tear": 6}},
            {"when": {"exam.lachman": "yes_firm_endpoint"}, "add": {"acl_tear": 3}},
            {"when": {"exam.pivot_shift": True}, "add": {"acl_tear": 4}},
            {"when": {"exam.posterior_drawer": True}, "add": {"pcl_tear": 6}},
            {"when": {"exam.mcmurray_medial": True}, "add": {"medial_meniscal_tear": 4}},
            {"when": {"exam.mcmurray_lateral": True}, "add": {"lateral_meniscal_tear": 4}},
            {"when": {"exam.joint_line_tenderness_medial": True}, "add": {"medial_meniscal_tear": 2}},
            {"when": {"exam.joint_line_tenderness_lateral": True}, "add": {"lateral_meniscal_tear": 2}},
            {"when": {"exam.mcl_laxity_grade": "grade1"}, "add": {"mcl_sprain": 2}},
            {"when": {"exam.mcl_laxity_grade": "grade2"}, "add": {"mcl_sprain": 3}},
            {"when": {"exam.mcl_laxity_grade": "grade3"}, "add": {"mcl_sprain": 4}},
            {"when": {"exam.lcl_laxity_grade": "grade1"}, "add": {"lcl_sprain": 2}},
            {"when": {"exam.lcl_laxity_grade": "grade2"}, "add": {"lcl_sprain": 3}},
            {"when": {"exam.lcl_laxity_grade": "grade3"}, "add": {"lcl_sprain": 4}},
            {"when": {"exam.pf_compression_clarke": True}, "add": {"patellofemoral_oa": 3, "pfps": 2}},
            {"when": {"exam.patellar_apprehension": True}, "add": {"patellar_instability": 4}},
            {"when": {"exam.j_sign": True}, "add": {"patellar_instability": 3}},
            {"when": {"exam.patella_alta": True}, "add": {"patellar_instability": 2}},
            {"when": {"exam.pf_crepitus": "chondral"}, "add": {"patellofemoral_oa": 2, "pfps": 2}},
            {"when": {"exam.pf_crepitus": "bone"}, "add": {"patellofemoral_oa": 3}},
            {"when": {"exam.alignment": "varus"}, "add": {"tibiofemoral_oa": 3}},
            {"when": {"exam.alignment": "valgus"}, "add": {"tibiofemoral_oa": 3}},
            {"when": {"exam.effusion": "mild"}, "add": {"acl_tear": 1, "pcl_tear": 1, "medial_meniscal_tear": 1, "lateral_meniscal_tear": 1}},
            {"when": {"exam.effusion": "moderate"}, "add": {"acl_tear": 2, "pcl_tear": 2, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2}},
            {"when": {"exam.effusion": "severe"}, "add": {"acl_tear": 3, "pcl_tear": 2, "medial_meniscal_tear": 2, "lateral_meniscal_tear": 2}},
            {"when": {"exam.rom_restriction": True}, "add": {"tibiofemoral_oa": 1, "medial_meniscal_tear": 1, "lateral_meniscal_tear": 1}},
            {"when": {"exam.fixed_flexion_deformity": True}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"exam.quads_tone": "reduced"}, "add": {"tibiofemoral_oa": 1, "pfps": 1}},
            {"when": {"exam.bakers_pseudocyst": True}, "add": {"bakers_cyst": 4}}
        ],
        "imaging": [
            {"when": {"imaging.xray_oa_tf": True}, "add": {"tibiofemoral_oa": 6}},
            {"when": {"imaging.xray_oa_pf": True}, "add": {"patellofemoral_oa": 4}},
            {"when": {"imaging.malalignment": "varus"}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"imaging.malalignment": "valgus"}, "add": {"tibiofemoral_oa": 2}},
            {"when": {"imaging.mri_acl": True}, "add": {"acl_tear": 8}},
            {"when": {"imaging.mri_pcl": True}, "add": {"pcl_tear": 8}},
            {"when": {"imaging.mri_medial_meniscus": True}, "add": {"medial_meniscal_tear": 8}},
            {"when": {"imaging.mri_lateral_meniscus": True}, "add": {"lateral_meniscal_tear": 8}}
        ]
    },
    "ranking": {
        "top_k": 3,
        "tie_breakers": [
            {"rule": "prefer_exam_positive_over_symptom_only"},
            {"rule": "prefer_mechanism_consistent"},
            {"rule": "prefer_side_specific_over_general"}
        ],
        "justification": {"max_reasons_per_dx": 4, "reason_selection": "highest_weighted_features"}
    },
    "output": {
        "fields": ["diagnosis_code", "score", "confidence_band", "key_drivers", "safety_net"],
        "confidence_bands": [
            {"min": 0, "max": 7, "label": "low"},
            {"min": 8, "max": 15, "label": "moderate"},
            {"min": 16, "max": 1000, "label": "high"}
        ],
        "safety_net_rules": [
            {"if": "any_red_flag_triggered", "message": "Urgent same-day assessment recommended."},
            {"if": "diagnosis_includes:painful_arthroplasty", "message": "Consider infection screen (CRP/ESR), targeted imaging, and arthroplasty review."}
        ]
    },
    "nlp_maps": {
        "mechanism_keywords": {
            "twisting": ["twist", "twisting", "pivot", "turned", "change of direction"],
            "pivot": ["pivot", "cutting"],
            "direct_blow": ["collision", "tackle", "blow", "contact", "fall onto knee", "dashboard"],
            "non_contact_jump_land": ["jump", "landing", "awkward landing"],
            "overuse": ["overuse", "gradual", "insidious", "training load", "running"],
            "post_op": ["post op", "after surgery", "postoperative"]
        },
        "knee_score_maxima": {
            "limp": 5, "pain": 25, "support": 5, "swelling": 10, "locking": 15,
            "stair_climbing": 10, "instability": 25, "squatting": 5
        }
    }
}

# Questionnaire form definitions
QUESTIONNAIRE_FORMS = {
    "knee_oa": {
        "name": "Knee Osteoarthritis Assessment",
        "spec": KNEE_OA_SPEC,
        "questions": [
            "What is your age?",
            "Which knee is affected? (left/right)",
            "How long have you had this problem? (acute/subacute/chronic)",
            "How did this problem start? (mechanism of injury)",
            "What type of pain do you experience? (sharp, dull, aching, burning)",
            "Does the pain radiate anywhere?",
            "Do you have any other symptoms like swelling, stiffness, or numbness?",
            "Is the pain constant or does it come and go?",
            "What makes the pain better or worse?",
            "On a scale of 0-10, how severe is your pain?",
            "Do you have morning stiffness? If so, how long does it last?",
            "Do you have stiffness after rest?",
            "How does this affect your daily activities? (walking, stairs, getting up from chair, etc.)",
            "Have you tried any treatments before?",
            "Do you have any red flag symptoms like fever, weight loss, or severe weakness?"
        ]
    },
    "knee_injury": {
        "name": "Knee Injury Assessment",
        "spec": KNEE_INJURY_SPEC,
        "questions": [
            "What is your age?",
            "Which knee is affected? (left/right)",
            "When did the injury occur?",
            "What activity were you doing when injured?",
            "How did the injury happen? (mechanism)",
            "What treatments have you tried?",
            "Have you had previous knee injuries or surgery?",
            "How is this affecting your activities?",
            "Do you have any red flag symptoms like fever, weight loss, or severe weakness?",
            "Please rate your knee function:",
            "  - Limp (none=5, slight=3, severe=0)",
            "  - Pain (none=25, slight=20, moderate=15, severe=10, extreme=5, unbearable=0)",
            "  - Support needed (none=5, cane=2, crutches=0)",
            "  - Swelling (none=10, slight=6, moderate=2, severe=0)",
            "  - Locking (none=15, catching=10, locked=6, frequently locked=2, locked rigid=0)",
            "  - Stair climbing (normal=10, slight=6, moderate=2, severe=0)",
            "  - Instability (none=25, slight=20, moderate=15, severe=10, extreme=5, severe=0)",
            "  - Squatting (normal=5, slight=4, moderate=2, severe=0)"
        ]
    }
}

def get_questionnaire_form(form_name: str) -> dict:
    """Get questionnaire form definition by name."""
    return QUESTIONNAIRE_FORMS.get(form_name, {})

def get_available_forms() -> list:
    """Get list of available questionnaire forms."""
    return list(QUESTIONNAIRE_FORMS.keys())

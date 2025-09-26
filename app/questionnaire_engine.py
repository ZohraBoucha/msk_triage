"""
Questionnaire Evaluation Engine for MSK Triage System

This module provides a reusable engine for evaluating questionnaire-based assessments
and generating differential diagnoses based on JSON specifications.
"""

from typing import Any, Dict, List, Tuple, Optional, Union
from copy import deepcopy
from pathlib import Path
import json


def get_by_path(d: Dict[str, Any], dotted: str) -> Any:
    """Get value from nested dictionary using dot notation."""
    cur = d
    for part in dotted.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def condition_match(input_obj: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """Match conditions for scoring rules."""
    for key, expected in when.items():
        val = get_by_path(input_obj, key)
        if key == 'phenotype' and isinstance(expected, str):
            if not isinstance(val, list) or expected not in val:
                return False
            continue
        if isinstance(expected, list):
            if val not in expected:
                return False
        elif isinstance(expected, str) and expected.startswith(('>', '<', '=')):
            try:
                num = float(val) if val is not None else None
            except Exception:
                return False
            if expected.startswith('>='):
                if not (num is not None and num >= float(expected[2:])):
                    return False
            elif expected.startswith('<='):
                if not (num is not None and num <= float(expected[2:])):  # correct slicing
                    return False
            elif expected.startswith('>'):
                if not (num is not None and num > float(expected[1:])):
                    return False
            elif expected.startswith('<'):
                if not (num is not None and num < float(expected[1:])):
                    return False
            elif expected.startswith('='):
                if not (num is not None and num == float(expected[1:])):
                    return False
        else:
            if val != expected:
                return False
    return True


def add_points(scores: Dict[str, int], reasons: Dict[str, List[Tuple[str, int]]], 
               addmap: Dict[str, int], reason_label: str):
    """Add points to diagnosis scores and track reasons."""
    for dx, pts in addmap.items():
        scores[dx] = scores.get(dx, 0) + int(pts)
        reasons.setdefault(dx, []).append((reason_label, int(pts)))


def apply_aggregate(rule: Dict[str, Any], input_obj: Dict[str, Any], 
                   scores: Dict[str, int], reasons: Dict[str, List[Tuple[str, int]]]):
    """Apply aggregate scoring rules."""
    for agg in rule.get('aggregate', []):
        method = agg.get('method')
        mapping = agg.get('map', {})
        
        if method == 'sum_function_items':
            func = get_by_path(input_obj, 'oa_index.function') or {}
            total = sum(mapping.get(v, 0) for v in func.values())
            for dx, expr in agg.get('then_add', {}).items():
                if expr.startswith('floor(total/'):
                    denom = int(expr[len('floor(total/'):-1])
                    pts = total // denom
                    if pts:
                        scores[dx] = scores.get(dx, 0) + pts
                        reasons.setdefault(dx, []).append(('Function difficulty aggregate', pts))
        
        elif method == 'sum_pf_loaded_items':
            items = agg.get('items', [])
            func = get_by_path(input_obj, 'oa_index.function') or {}
            total = sum(mapping.get(func.get(item, 'none'), 0) for item in items)
            for dx, expr in agg.get('then_add', {}).items():
                if expr.startswith('floor(total/'):
                    denom = int(expr[len('floor(total/'):-1])
                    pts = total // denom
                    if pts:
                        scores[dx] = scores.get(dx, 0) + pts
                        reasons.setdefault(dx, []).append(('PF-loaded tasks aggregate', pts))
        
        elif method == 'deficit':
            field = agg.get('field')
            max_val = agg.get('max', 0)
            scale = agg.get('scale', 1.0)
            
            # Get the actual value for the field
            field_value = get_by_path(input_obj, f'knee_score.{field}') or 0
            deficit = max_val - field_value
            scaled_deficit = round(scale * deficit)
            
            for dx, expr in agg.get('then_add', {}).items():
                if expr.startswith('round('):
                    # Parse expressions like "round(scale*deficit)" or "round(0.8*scale*deficit)"
                    expr_clean = expr[6:-1]  # Remove "round(" and ")"
                    if 'scale*deficit' in expr_clean:
                        multiplier = 1.0
                        if expr_clean != 'scale*deficit':
                            # Extract multiplier like "0.8*scale*deficit"
                            multiplier = float(expr_clean.split('*')[0])
                        pts = round(multiplier * scaled_deficit)
                    else:
                        pts = scaled_deficit
                    
                    if pts > 0:
                        scores[dx] = scores.get(dx, 0) + pts
                        reasons.setdefault(dx, []).append((f'{field} deficit scoring', pts))
        
        elif method == 'total':
            # Calculate total knee score and apply scaling
            knee_score = get_by_path(input_obj, 'knee_score') or {}
            total = sum(knee_score.values()) if knee_score else 0
            scale = agg.get('scale', 0.1)
            
            for dx, expr in agg.get('then_add', {}).items():
                if expr.startswith('round(scale*('):
                    # Parse expressions like "round(scale*(100-total))"
                    expr_clean = expr[6:-1]  # Remove "round(" and ")"
                    if '100-total' in expr_clean:
                        pts = round(scale * (100 - total))
                        if pts > 0:
                            scores[dx] = scores.get(dx, 0) + pts
                            reasons.setdefault(dx, []).append(('Total knee score aggregate', pts))


def run_questionnaire_engine(spec: Dict[str, Any], input_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the questionnaire evaluation engine.
    
    Args:
        spec: JSON specification for the questionnaire
        input_obj: Patient input data
    
    Returns:
        Dictionary with diagnosis results, confidence bands, and safety net messages
    """
    # 1. Check red flags first
    for rf in spec.get('red_flag_logic', []):
        if_all = rf.get('if_all_true', [])
        if all(bool(get_by_path(input_obj, path)) for path in if_all):
            action = rf['action']
            return {
                'route': 'urgent',
                'urgent_reason': if_all,
                'provisional_diagnosis': action.get('diagnosis'),
                'message': 'Urgent same-day assessment recommended.'
            }
    
    # 2. Initialize scores
    dx_codes = spec['diagnoses']
    scores = {dx: 0 for dx in dx_codes}
    reasons: Dict[str, List[Tuple[str, int]]] = {}
    
    # 3. Apply scoring blocks
    blocks = spec.get('scoring', {})
    
    # Apply scoring in order: mechanism, symptoms, oa_index, exam, imaging
    for block_name in ['mechanism', 'onset_mechanism', 'symptoms', 'oa_index', 'knee_score', 
                      'symptoms_from_text', 'exam', 'imaging']:
        for rule in blocks.get(block_name, []):
            when = rule.get('when', {})
            
            # Handle special cases for aggregate rules
            if 'aggregate' in rule:
                if get_by_path(input_obj, 'oa_index') is not None or get_by_path(input_obj, 'knee_score') is not None:
                    apply_aggregate(rule, input_obj, scores, reasons)
            else:
                if condition_match(input_obj, when):
                    if 'add' in rule:
                        add_points(scores, reasons, rule['add'], f"{block_name}:{when}")
                    if 'add_all' in rule:
                        addmap = {dx: int(rule['add_all']) for dx in dx_codes}
                        add_points(scores, reasons, addmap, f"{block_name}:{when}")
    
    # 4. Rank results
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_k = spec.get('ranking', {}).get('top_k', 3)
    
    # 5. Apply confidence bands
    bands = spec.get('output', {}).get('confidence_bands', [])
    def band(score: int) -> str:
        for b in bands:
            if b['min'] <= score <= b['max']:
                return b['label']
        return 'unknown'
    
    # 6. Generate safety net messages
    safety_msgs = []
    sn_rules = spec.get('output', {}).get('safety_net_rules', [])
    any_red_flag = any(bool(get_by_path(input_obj, path))
                       for path in ['red_flags.fever_unwell_hot_joint', 
                                   'red_flags.true_locked_knee', 
                                   'red_flags.inability_slr_after_eccentric_load'])
    
    for sn in sn_rules:
        cond = sn.get('if', '')
        if cond == 'any_red_flag_triggered' and any_red_flag:
            safety_msgs.append(sn['message'])
    
    # 7. Build results
    results = []
    for dx, sc in ranked[:top_k]:
        if sc == 0:  # Skip diagnoses with no points
            continue
        
        # Get key drivers (top reasons for this diagnosis)
        rlist = sorted(reasons.get(dx, []), key=lambda t: t[1], reverse=True)
        max_reasons = spec.get('ranking', {}).get('justification', {}).get('max_reasons_per_dx', 4)
        key_drivers = [f"{lbl} (+{pts})" for lbl, pts in rlist[:max_reasons]]
        
        results.append({
            'diagnosis_code': dx,
            'score': sc,
            'confidence_band': band(sc),
            'key_drivers': key_drivers
        })
        
        # Add conditional safety net messages
        for sn in sn_rules:
            if sn.get('if', '').startswith('diagnosis_includes:'):
                target = sn['if'].split(':', 1)[1]
                if dx == target and sn['message'] not in safety_msgs:
                    safety_msgs.append(sn['message'])
    
    return {
        'route': 'routine',
        'top': results,
        'safety_net': safety_msgs
    }


def load_questionnaire_spec(spec_path: str) -> Dict[str, Any]:
    """Load questionnaire specification from JSON file."""
    with open(spec_path, 'r') as f:
        return json.load(f)


def map_mechanism_from_text(mechanism_text: str, spec: Dict[str, Any]) -> str:
    """Map free text mechanism description to standardized enum."""
    if not mechanism_text:
        return 'unknown'
    
    mechanism_text = mechanism_text.lower()
    keywords = spec.get('nlp_maps', {}).get('mechanism_keywords', {})
    
    for mechanism, keyword_list in keywords.items():
        if any(keyword in mechanism_text for keyword in keyword_list):
            return mechanism
    
    return 'unknown'


def calculate_knee_score_total(knee_score: Dict[str, int]) -> int:
    """Calculate total knee score from individual components."""
    return sum(knee_score.values()) if knee_score else 0


def get_diagnosis_display_name(diagnosis_code: str) -> str:
    """Convert diagnosis code to human-readable name."""
    display_names = {
        'tibiofemoral_oa': 'Tibiofemoral Osteoarthritis',
        'patellofemoral_oa': 'Patellofemoral Osteoarthritis',
        'pfps': 'Patellofemoral Pain Syndrome',
        'acl_tear': 'Anterior Cruciate Ligament Tear',
        'pcl_tear': 'Posterior Cruciate Ligament Tear',
        'medial_meniscal_tear': 'Medial Meniscal Tear',
        'lateral_meniscal_tear': 'Lateral Meniscal Tear',
        'mcl_sprain': 'Medial Collateral Ligament Sprain',
        'lcl_sprain': 'Lateral Collateral Ligament Sprain',
        'patellar_instability': 'Patellar Instability',
        'painful_arthroplasty': 'Painful Arthroplasty',
        'bakers_cyst': "Baker's Cyst",
        'loose_body': 'Loose Body',
        'septic_arthritis': 'Septic Arthritis',
        'bucket_handle_meniscal_tear': 'Bucket Handle Meniscal Tear',
        'extensor_mechanism_rupture': 'Extensor Mechanism Rupture'
    }
    return display_names.get(diagnosis_code, diagnosis_code.replace('_', ' ').title())

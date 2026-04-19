"""
What-If Simulation API Routes
================================
Batch mutation analysis and positional scanning for
"what if we mutated here?" exploration mode.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    WhatIfBatchRequest, WhatIfScanRequest, 
    WhatIfScanResult, WhatIfScanResponse,
    MutationSimulationResponse,
)
from app.services.biology_engine import perform_quality_control, simulate_translation
from app.services.mutation_engine import apply_mutation, annotate_variant
from app.services.ai_predictors import (
    calculate_grantham_score, predict_stability_ddg,
    predict_functional_risk, aggregate_structural_risk,
)
from typing import List

router = APIRouter()

BASES = ['A', 'T', 'G', 'C']


@router.post("/scan", response_model=WhatIfScanResponse)
async def whatif_scan(request: WhatIfScanRequest):
    """
    Positional scanning: for each position in range, try mutating to every other base.
    Returns a heatmap-ready matrix of impact scores.
    Uses fast rule-based analysis (no ESM) for scan speed.
    """
    sequence = request.sequence.upper().strip()
    qc = perform_quality_control(sequence)
    if not qc.is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid sequence: {qc.errors}")
    
    start = request.start_position
    end = request.end_position if request.end_position is not None else len(sequence)
    end = min(end, len(sequence))
    
    # Limit scan range to prevent excessive computation
    if end - start > 120:
        end = start + 120
    
    baseline_translation = simulate_translation(sequence)
    wt_protein = baseline_translation.amino_acid_sequence
    
    scan_results: List[WhatIfScanResult] = []
    
    for pos in range(start, end):
        original_base = sequence[pos]
        
        for target_base in BASES:
            if target_base == original_base:
                continue
            
            try:
                mutated_seq = apply_mutation(sequence, "point", pos, target_base)
                mut_translation = simulate_translation(mutated_seq)
                mut_protein = mut_translation.amino_acid_sequence
                
                from app.services.mutation_engine import detect_frameshift
                is_fs = detect_frameshift("point", target_base)
                variant = annotate_variant(wt_protein, mut_protein, is_fs)
                
                # Fast heuristic scoring (no ESM for scan speed)
                codon_pos = pos // 3
                wt_aa = wt_protein[codon_pos] if codon_pos < len(wt_protein) else ""
                mut_aa = mut_protein[codon_pos] if codon_pos < len(mut_protein) else ""
                
                grantham = calculate_grantham_score(wt_aa, mut_aa)
                ddg = predict_stability_ddg(sequence, mutated_seq, wt_aa, mut_aa)
                risk = predict_functional_risk(variant, ddg)
                
                # Embedding risk approximation from grantham
                emb_risk = min(1.0, grantham / 200.0) if variant != "Silent" else 0.0
                
                scan_results.append(WhatIfScanResult(
                    position=pos,
                    original_base=original_base,
                    mutated_base=target_base,
                    variant_annotation=variant,
                    functional_risk=risk,
                    stability_score=ddg,
                    embedding_risk_score=round(emb_risk, 3),
                ))
            except Exception:
                continue
    
    return WhatIfScanResponse(
        scan_results=scan_results,
        sequence_length=len(sequence),
        scan_range=[start, end],
        status="SUCCEEDED",
    )


@router.post("/batch")
async def whatif_batch(request: WhatIfBatchRequest):
    """
    Apply multiple mutations sequentially and return impact for each.
    Useful for testing mutation combinations.
    """
    sequence = request.sequence.upper().strip()
    qc = perform_quality_control(sequence)
    if not qc.is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid sequence: {qc.errors}")
    
    if len(request.mutations) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 mutations per batch")
    
    baseline_translation = simulate_translation(sequence)
    wt_protein = baseline_translation.amino_acid_sequence
    
    results = []
    
    for mut in request.mutations:
        try:
            mut_type = mut.get("mutation_type", "point")
            position = mut.get("position", 0)
            new_nuc = mut.get("new_nucleotide")
            
            mutated_seq = apply_mutation(sequence, mut_type, position, new_nuc)
            mut_translation = simulate_translation(mutated_seq)
            mut_protein = mut_translation.amino_acid_sequence
            
            from app.services.mutation_engine import detect_frameshift
            is_fs = detect_frameshift(mut_type, new_nuc)
            variant = annotate_variant(wt_protein, mut_protein, is_fs)
            
            codon_pos = position // 3
            wt_aa = wt_protein[codon_pos] if codon_pos < len(wt_protein) else ""
            mut_aa = mut_protein[codon_pos] if codon_pos < len(mut_protein) else ""
            
            grantham = calculate_grantham_score(wt_aa, mut_aa)
            ddg = predict_stability_ddg(sequence, mutated_seq, wt_aa, mut_aa)
            risk = predict_functional_risk(variant, ddg)
            
            results.append({
                "mutation": mut,
                "mutated_sequence": mutated_seq,
                "variant_annotation": variant,
                "functional_risk": risk,
                "stability_score": ddg,
                "grantham_score": grantham,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "mutation": mut,
                "error": str(e),
                "status": "error"
            })
    
    return {"results": results, "count": len(results), "status": "SUCCEEDED"}

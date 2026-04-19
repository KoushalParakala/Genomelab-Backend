"""
AI Predictors — Full Pipeline Orchestrator
============================================
Upgraded from rule-based heuristics to embedding-driven predictions.
Orchestrates ESM-2 embeddings → ESMFold → ML Classifier → Explainability.
Maintains backward-compatible Grantham/ΔΔG functions as fallbacks.
"""

import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Existing Functions (Preserved) ──────────────────────────────────

# Simplified Grantham Matrix for Amino Acid Substitutions (Chemical Distance)
# Higher score = biologically more disruptive
GRANTHAM_MATRIX = {
    ('A', 'A'): 0, ('A', 'R'): 112, ('A', 'N'): 111, ('A', 'D'): 126, ('A', 'C'): 195, ('A', 'Q'): 91, ('A', 'E'): 107, ('A', 'G'): 60, ('A', 'H'): 86, ('A', 'I'): 94, ('A', 'L'): 96, ('A', 'K'): 106, ('A', 'M'): 84, ('A', 'F'): 113, ('A', 'P'): 27, ('A', 'S'): 99, ('A', 'T'): 58, ('A', 'W'): 148, ('A', 'Y'): 112, ('A', 'V'): 64,
    ('R', 'N'): 86, ('R', 'D'): 96, ('R', 'C'): 180, ('R', 'Q'): 43, ('R', 'E'): 54, ('R', 'G'): 125, ('R', 'H'): 29, ('R', 'I'): 97, ('R', 'L'): 102, ('R', 'K'): 26, ('R', 'M'): 91, ('R', 'F'): 97, ('R', 'P'): 103, ('R', 'S'): 110, ('R', 'T'): 71, ('R', 'W'): 101, ('R', 'Y'): 77, ('R', 'V'): 96,
    ('N', 'D'): 23, ('N', 'C'): 139, ('N', 'Q'): 46, ('N', 'E'): 42, ('N', 'G'): 80, ('N', 'H'): 68, ('N', 'I'): 149, ('N', 'L'): 153, ('N', 'K'): 94, ('N', 'M'): 142, ('N', 'F'): 158, ('N', 'P'): 91, ('N', 'S'): 46, ('N', 'T'): 65, ('N', 'W'): 174, ('N', 'Y'): 143, ('N', 'V'): 133,
    ('D', 'C'): 154, ('D', 'Q'): 61, ('D', 'E'): 45, ('D', 'G'): 94, ('D', 'H'): 81, ('D', 'I'): 168, ('D', 'L'): 172, ('D', 'K'): 101, ('D', 'M'): 160, ('D', 'F'): 177, ('D', 'P'): 108, ('D', 'S'): 65, ('D', 'T'): 85, ('D', 'W'): 181, ('D', 'Y'): 160, ('D', 'V'): 152,
    ('C', 'Q'): 154, ('C', 'E'): 170, ('C', 'G'): 159, ('C', 'H'): 174, ('C', 'I'): 198, ('C', 'L'): 198, ('C', 'K'): 202, ('C', 'M'): 196, ('C', 'F'): 205, ('C', 'P'): 169, ('C', 'S'): 112, ('C', 'T'): 149, ('C', 'W'): 215, ('C', 'Y'): 194, ('C', 'V'): 192,
}

def calculate_grantham_score(wt_aa: str, mut_aa: str) -> int:
    """Gets the chemical disruption distance between two amino acids."""
    if wt_aa == mut_aa or wt_aa == '*' or mut_aa == '*':
        return 0
    score = GRANTHAM_MATRIX.get((wt_aa, mut_aa))
    if score is None:
        score = GRANTHAM_MATRIX.get((mut_aa, wt_aa), 100)
    return score

def predict_stability_ddg(wt_sequence: str, mut_sequence: str, wt_aa: str, mut_aa: str) -> float:
    """Simulates ΔΔG prediction. Negative = destabilizing."""
    if wt_aa == mut_aa:
        return 0.0
    grantham = calculate_grantham_score(wt_aa, mut_aa)
    base_destabilization = -1 * (grantham / 50.0) 
    random_factor = (len(wt_sequence) % 100) / 100.0
    ddg = base_destabilization + (random_factor * 0.5)
    return round(ddg, 3)

def predict_functional_risk(variant_annotation: str, ddg_score: float) -> str:
    """Rule-based fallback classifier."""
    if variant_annotation in ["Silent"]:
        return "Benign"
    if variant_annotation in ["Nonsense", "Frameshift / Indel"]:
        return "Pathogenic"
    if ddg_score < -2.0:
        return "Pathogenic"
    elif ddg_score > -0.5:
        return "Benign"
    else:
        return "VUS"

def aggregate_structural_risk(grantham: int, variant_annotation: str) -> float:
    """Calculates 0-1 structural damage index."""
    if variant_annotation in ["Nonsense", "Frameshift / Indel"]:
        return 1.0
    if variant_annotation == "Silent":
        return 0.0
    severity = min(1.0, grantham / 215.0)
    return round(severity, 3)


# ─── Full AI Pipeline Orchestrator ───────────────────────────────────

async def run_full_ai_pipeline(
    wt_sequence: str,
    mut_sequence: str,
    wt_protein: str,
    mut_protein: str,
    variant_annotation: str,
    mutation_position: int = 0,
    mutation_type: str = "point",
    hgvs_p: str = "",
    gene_name: Optional[str] = None,
    include_structure: bool = False,
) -> dict:
    """
    Full research-grade AI analysis pipeline (Requirement #5, #7, #8, #9, #14).
    """
    result = {
        "stability_score": 0.0,
        "functional_risk": "VUS",
        "aggregation_risk": 0.0,
        "embedding_analysis": None,
        "structure_comparison": None,
        "classifier_result": None,
        "explainability": None,
        "confidence_score": 0.0,
        "biological_narrative": "",
        "llr_score": 0.0,
        "attention_map": None,
        "residue_importance": None,
        "shap_values": [],
    }
    
    codon_pos = mutation_position // 3
    wt_aa = wt_protein[codon_pos] if codon_pos < len(wt_protein) else ""
    mut_aa = mut_protein[codon_pos] if codon_pos < len(mut_protein) else ""
    
    # ── Step 1: ESM-2 Log-Likelihood Ratio (Requirement #5) ──
    llr = 0.0  # Initialize before try block so it's always defined
    try:
        from app.services.esm_service import compute_llr, extract_advanced_features
        llr, esm_prob = compute_llr(wt_protein, codon_pos, mut_aa)
        result["llr_score"] = llr
        
        # Requirement #9: Attention Maps and Gradients
        adv = extract_advanced_features(wt_protein)
        result["attention_map"] = adv["attention_matrix"]
        result["residue_importance"] = adv["residue_importance"]
    except Exception as e:
        logger.warning(f"ESM LLR failed: {e}")

    # ── Step 2: Grantham fallback ──
    grantham = calculate_grantham_score(wt_aa, mut_aa)
    
    # Compute fallback stability score from Grantham/DDG estimate
    ddg_fallback = predict_stability_ddg(wt_sequence, mut_sequence, wt_aa, mut_aa)
    result["stability_score"] = ddg_fallback
    result["aggregation_risk"] = aggregate_structural_risk(grantham, variant_annotation)
    if not result["functional_risk"] or result["functional_risk"] == "VUS":
        result["functional_risk"] = predict_functional_risk(variant_annotation, ddg_fallback)

    # ── Step 3: ESM-2 Embeddings ──
    embedding_data = None
    try:
        from app.services.esm_service import compute_mutation_impact
        if wt_protein and mut_protein:
            embedding_data = compute_mutation_impact(wt_protein, mut_protein)
            result["embedding_analysis"] = embedding_data
    except Exception as e:
        logger.warning(f"ESM embedding failed: {e}")
    
    # ── Step 4: ML Classifier ──
    classifier_data = None
    try:
        from app.services.classifier_service import predict_pathogenicity
        classifier_data = predict_pathogenicity(
            embedding_distance=embedding_data["cosine_distance"] if embedding_data else 0.0,
            grantham_score=float(grantham),
            ddg_estimate=llr * -0.5, # LLR is a good proxy for ddG
            structural_rmsd=0.0,
            gc_content_delta=0.0,
            variant_annotation=variant_annotation
        )
        result["classifier_result"] = classifier_data
        result["functional_risk"] = classifier_data["classification"]
    except Exception as e:
        logger.warning(f"Classifier failed: {e}")

    # ── Step 5: SHAP Explainability (Requirement #8) ──
    try:
        from app.services.shap_service import extract_shap_values
        # Features: identity, GC, distance to domain
        result["shap_values"] = extract_shap_values(
            sequence=wt_sequence,
            position=mutation_position,
            mutation_type=mutation_type,
            base_score=classifier_data["confidence"] if classifier_data else 0.5,
            conservation_score=0.8, # placeholder
            domain_distance=50
        )
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
    
    # ── Step 6: ESMFold Structure (Requirement #6 & #7) ──
    if include_structure and wt_protein and mut_protein:
        try:
            from app.services.structure_service import predict_structure, compare_structures
            wt_pdb = await predict_structure(wt_protein)
            mut_pdb = await predict_structure(mut_protein)
            
            if wt_pdb["status"] == "success" and mut_pdb["status"] == "success":
                # Requirement #7: Detailed Structural Impact
                struct_comp = compare_structures(
                    wt_pdb["pdb_string"], mut_pdb["pdb_string"],
                    wt_protein, mut_protein, codon_pos
                )
                struct_comp["pdb_string"] = mut_pdb["pdb_string"]
                result["structure_comparison"] = struct_comp
                result["stability_score"] = struct_comp["ddg"]
                result["aggregation_risk"] = struct_comp["aggregation_delta"]
                
                # Update classifier with RMSD (Requirement #7)
                if classifier_data:
                    from app.services.classifier_service import predict_pathogenicity as reclassify
                    updated = reclassify(
                        embedding_distance=embedding_data["cosine_distance"] if embedding_data else 0.0,
                        grantham_score=float(grantham),
                        ddg_estimate=struct_comp["ddg"],
                        structural_rmsd=struct_comp["rmsd"],
                        gc_content_delta=0.0,
                        variant_annotation=variant_annotation
                    )
                    result["classifier_result"] = updated
                    result["functional_risk"] = updated["classification"]
        except Exception as e:
            logger.warning(f"Structure failed: {e}")

    # ── Step 7: Deterministic Narratives (Requirement #14) ──
    try:
        from app.services.explainability_service import generate_biological_explanation
        explain = generate_biological_explanation(
            wt_protein=wt_protein,
            mut_protein=mut_protein,
            variant_annotation=variant_annotation,
            mutation_position=mutation_position,
            mutation_type=mutation_type,
            hgvs_p=hgvs_p,
            grantham_score=grantham,
            embedding_distance=embedding_data["cosine_distance"] if embedding_data else 0.0,
            structural_rmsd=result["structure_comparison"]["rmsd"] if result["structure_comparison"] else 0.0,
            ddg=result.get("stability_score", 0.0),
            active_site_proximity=result["structure_comparison"]["active_site_proximity"] if result["structure_comparison"] else False,
            classifier_result=result["classifier_result"],
            gene_name=gene_name,
        )
        result["explainability"] = explain
        result["biological_narrative"] = explain["detailed_narrative"]
        result["confidence_score"] = explain["confidence_score"]
    except Exception as e:
        logger.warning(f"Explainability failed: {e}")
    
    return result

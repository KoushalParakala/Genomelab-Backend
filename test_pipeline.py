"""Quick test script for ESM service."""
import sys
sys.path.insert(0, ".")

from app.services.esm_service import ensure_model_loaded, compute_mutation_impact

print("Loading ESM model...")
ensure_model_loaded()
print("ESM model loaded!")

print("\nTesting mutation impact: MVHLTPEEK -> MVHLTVEEK")
result = compute_mutation_impact("MVHLTPEEK", "MVHLTVEEK")
print(f"  Cosine distance: {result['cosine_distance']:.6f}")
print(f"  Euclidean distance: {result['euclidean_distance']:.4f}")
print(f"  Embedding risk score: {result['embedding_risk_score']:.4f}")
print(f"  Max residue shift: {result['max_residue_shift']:.4f}")
print(f"  Affected region: {result['affected_region']}")

print("\nTesting classifier...")
from app.services.classifier_service import ensure_classifier_loaded, predict_pathogenicity
ensure_classifier_loaded()
cls_result = predict_pathogenicity(
    embedding_distance=result['cosine_distance'],
    grantham_score=100,
    ddg_estimate=-2.0,
    structural_rmsd=0.0,
    gc_content_delta=0.0,
    variant_annotation="Missense"
)
print(f"  Classification: {cls_result['classification']}")
print(f"  Confidence: {cls_result['confidence']:.3f}")
print(f"  Probabilities: {cls_result['probabilities']}")

print("\nTesting explainability...")
from app.services.explainability_service import generate_biological_explanation
explain = generate_biological_explanation(
    wt_protein="MVHLTPEEK",
    mut_protein="MVHLTVEEK",
    variant_annotation="Missense",
    mutation_position=15,
    mutation_type="point",
    grantham_score=100,
    embedding_distance=result['cosine_distance'],
    classifier_result=cls_result,
)
print(f"  Summary: {explain['summary']}")
print(f"  Confidence: {explain['confidence_level']} ({explain['confidence_score']:.3f})")
print(f"  Consequences: {explain['molecular_consequences'][:2]}")

print("\n=== ALL TESTS PASSED ===")

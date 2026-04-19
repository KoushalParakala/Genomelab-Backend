from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union

# ─── Existing Models (preserved for backward compatibility) ───────────

class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="The nucleotide sequence (DNA or RNA)")
    sequence_id: Optional[str] = Field(None, description="Optional identifier for the sequence")
    gene_name: Optional[str] = Field(None, description="Optional gene name (e.g. BRCA1, TP53)")
    
class MutationRequest(BaseModel):
    sequence: str = Field(..., description="The baseline sequence")
    mutation_type: str = Field(..., description="point, insertion, deletion")
    position: int = Field(..., description="0-indexed position of the mutation")
    new_nucleotide: Optional[str] = Field(None, description="Nucleotide to insert/replace")
    gene_name: Optional[str] = Field(None, description="Optional gene name for context")
    include_structure: bool = Field(False, description="Request ESMFold structure prediction")
    session_id: Optional[str] = Field(None, description="Requirement #11: Session ID for mutation replay series")
    baseline_id: Optional[str] = Field(None, description="Requirement #13: Link to baseline sequence record")

class QualityControlResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    length: int
    gc_content: float
    reading_frame: int = Field(0, description="Offset to the first AUG start codon (0, 1, or 2)")

class TranslationResult(BaseModel):
    mrna_sequence: str
    amino_acid_sequence: str
    stop_codon_detected: bool
    frame_shift_detected: bool

class SequenceSimulationResponse(BaseModel):
    qc: QualityControlResult
    translation: Optional[TranslationResult] = None
    status: str
    sequence_record_id: Optional[Union[int, str]] = None


# ─── Enhanced AI Models ──────────────────────────────────────────────

class ESMEmbeddingResult(BaseModel):
    """ESM-2 embedding analysis summary."""
    cosine_distance: float = Field(0.0, description="Cosine distance between WT and mutant embeddings")
    euclidean_distance: float = Field(0.0, description="Euclidean distance between mean embeddings")
    per_residue_delta: List[float] = Field(default_factory=list, description="Per-residue embedding shift magnitudes")
    max_residue_shift: float = Field(0.0, description="Maximum per-residue shift")
    affected_region: List[int] = Field(default_factory=lambda: [0, 0], description="Start/end of most affected residue range")
    embedding_risk_score: float = Field(0.0, description="Normalized embedding-based risk (0-1)")

class DetailedAnnotation(BaseModel):
    variant_type: str = Field(..., description="Silent, Missense, Nonsense, Frameshift / Indel")
    hgvs_c: str = Field("", description="HGVS DNA level notation")
    hgvs_p: str = Field("", description="HGVS protein level notation")
    codon_change: str = Field("", description="e.g. AAA -> GAA")
    amino_acid_change: str = Field("", description="e.g. Lys -> Glu")
    codon_affected: Optional[int] = Field(None, description="The index of the affected codon")
    stop_codon_introduced: bool = Field(False, description="True if a premature stop codon was introduced")
    reading_frame_preserved: bool = Field(True, description="True if the reading frame was not disrupted")
    domain_boundary: Optional[str] = Field(None, description="Known functional domain")
    conservation_score: float = Field(0.0, description="Evolutionary conservation (0-1)")
    high_risk_conservation: bool = Field(False, description="True if conservation >= 0.8")

class StructureComparisonResult(BaseModel):
    """Structural comparison between WT and mutant proteins (Requirement #7)."""
    rmsd: float = Field(0.0, description="CA-atom RMSD in Angstroms")
    ddg: float = Field(0.0, description="Estimated change in folding free energy (kcal/mol)")
    h_bonds_broken: int = Field(0, description="Count of hydrogen bonds present in WT but not Mutant")
    h_bonds_formed: int = Field(0, description="Count of hydrogen bonds present in Mutant but not WT")
    secondary_structure_diff: List[Dict[str, Any]] = Field(default_factory=list, description="Diff of Helix/Sheet/Coil assignments")
    active_site_proximity: bool = Field(False, description="True if mutation is within 8Å of a catalytic site")
    aggregation_delta: float = Field(0.0, description="Change in predicted aggregation propensity")
    
    # Existing displacement fields
    per_residue_displacement: List[float] = Field(default_factory=list, description="Per-residue displacement (Å)")
    max_displacement: float = Field(0.0, description="Maximum per-residue displacement")
    max_displacement_residue: int = Field(0, description="Residue with maximum displacement")
    mean_plddt_wt: float = Field(0.0, description="Mean pLDDT confidence for WT structure")
    mean_plddt_mut: float = Field(0.0, description="Mean pLDDT confidence for mutant structure")
    stability_assessment: str = Field("", description="Human-readable stability assessment")
    structural_risk_score: float = Field(0.0, description="Structural risk (0-1)")

class ClassifierResult(BaseModel):
    """ML classifier prediction output."""
    classification: str = Field(..., description="Pathogenic, Benign, or VUS")
    confidence: float = Field(..., description="Classifier confidence (0-1)")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Class probabilities")
    feature_importances: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")

class EvidenceSource(BaseModel):
    """A single evidence source and its verdict."""
    source: str
    verdict: str
    weight: float
    detail: str

class ExplainabilityResult(BaseModel):
    """Biological explanation and narrative."""
    summary: str = Field("", description="1-2 sentence summary")
    detailed_narrative: str = Field("", description="Full scientific explanation")
    molecular_consequences: List[str] = Field(default_factory=list)
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)
    confidence_level: str = Field("Low", description="High, Medium, or Low")
    confidence_score: float = Field(0.0, description="0-1 confidence score")
    gene_context: Optional[Dict[str, Any]] = None

class AIPredictions(BaseModel):
    """Enhanced AI predictions — backward compatible + new fields (Requirements #5, #8, #9)."""
    # Existing fields
    stability_score: float = Field(..., description="ΔΔG Stability Prediction")
    functional_risk: str = Field(..., description="Pathogenic, Benign, or VUS")
    aggregation_risk: float = Field(..., description="0.0 to 1.0 structural disruption index")
    
    # New high-fidelity fields
    llr_score: Optional[float] = Field(None, description="ESM-2 substitution log-likelihood ratio")
    attention_map: Optional[List[List[float]]] = Field(None, description="Final layer transformer attention weights")
    shap_values: List[Dict[str, Any]] = Field(default_factory=list, description="Sorted feature importance values")
    
    # Existing analysis sub-models
    embedding_analysis: Optional[ESMEmbeddingResult] = None
    structure_comparison: Optional[StructureComparisonResult] = None
    classifier_result: Optional[ClassifierResult] = None
    explainability: Optional[ExplainabilityResult] = None
    
    confidence_score: float = Field(0.0, description="Overall confidence (0-1)")
    biological_narrative: str = Field("", description="AI-generated explanation")

class MutationSimulationResponse(BaseModel):
    """Full research-grade simulation response (Requirements #11, #12)."""
    baseline_sequence: str
    mutated_sequence: str
    mutation_type: str
    position: int
    new_nucleotide: Optional[str] = None
    baseline_translation: Optional[TranslationResult] = None
    mutated_translation: Optional[TranslationResult] = None
    variant_annotation: str
    detailed_annotation: Optional[DetailedAnnotation] = None
    ai_predictions: AIPredictions
    
    # Mutation Replay State Series (Requirement #11)
    replay_series: List[Dict[str, Any]] = Field(default_factory=list, description="Ordered array of mutation snapshots")
    # Sequence Fingerprint (Requirement #12)
    fingerprint: List[int] = Field(default_factory=list, description="8-value radial waveform array")
    
    status: str
    log_id: Optional[Union[int, str]] = None


# ─── Structure Prediction Models ─────────────────────────────────────

class StructurePredictionRequest(BaseModel):
    protein_sequence: str = Field(..., description="Amino acid sequence")

class StructurePredictionResponse(BaseModel):
    pdb_string: str = Field("", description="PDB file content")
    sequence_length: int = 0
    status: str = "pending"
    error: Optional[str] = None

class StructureCompareRequest(BaseModel):
    wt_pdb: str = Field(..., description="Wild-type PDB string")
    mut_pdb: str = Field(..., description="Mutant PDB string")


# ─── What-If Simulation Models ──────────────────────────────────────

class WhatIfBatchRequest(BaseModel):
    """Batch mutation analysis — apply multiple mutations to a sequence."""
    sequence: str
    mutations: List[Dict[str, Any]] = Field(..., description="List of {mutation_type, position, new_nucleotide}")

class WhatIfScanRequest(BaseModel):
    """Positional scanning — mutate every position to every base."""
    sequence: str
    start_position: int = Field(0, description="Start of scan range (0-indexed)")
    end_position: Optional[int] = Field(None, description="End of scan range (defaults to full sequence)")

class WhatIfScanResult(BaseModel):
    """Result for a single position-base mutation in the scan."""
    position: int
    original_base: str
    mutated_base: str
    variant_annotation: str
    functional_risk: str
    stability_score: float
    embedding_risk_score: float = 0.0

class WhatIfScanResponse(BaseModel):
    """Full positional scan results."""
    scan_results: List[WhatIfScanResult]
    sequence_length: int
    scan_range: List[int]
    status: str


# ─── Experiment Models ───────────────────────────────────────────────

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    mutation_log_ids: List[Union[int, str]] = Field(default_factory=list)

class ExperimentResponse(BaseModel):
    id: Union[int, str]
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    mutation_count: int = 0

class SharedResultResponse(BaseModel):
    token: str
    expires_at: Optional[str] = None
    experiment_id: Union[int, str]

# ─── History & Archival Models ───────────────────────────────────────

class MutationLogSummary(BaseModel):
    id: str
    session_id: Optional[str]
    mutation_type: str
    position: int
    variant_annotation: str
    verdict: Optional[str]
    created_at: Any
    stability_score: Optional[float]

class MutationHistoryResponse(BaseModel):
    logs: List[MutationLogSummary]
    total_count: int
    status: str

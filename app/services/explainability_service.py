"""
Biological Explainability Service
====================================
Generates human-readable biological narratives explaining the impact of 
mutations. Maps variants to gene functions, describes molecular consequences,
and provides evidence-based confidence assessments.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Gene Function Database ──────────────────────────────────────────
# Curated dictionary of common clinically-relevant genes

GENE_DATABASE = {
    "BRCA1": {
        "full_name": "Breast Cancer Type 1 Susceptibility Protein",
        "function": "DNA repair via homologous recombination; tumor suppressor",
        "disease_associations": ["Breast cancer", "Ovarian cancer", "Prostate cancer"],
        "critical_domains": ["RING domain (1-109)", "BRCT domain (1646-1859)"],
        "conservation": "Highly conserved across mammals"
    },
    "TP53": {
        "full_name": "Tumor Protein P53",
        "function": "Transcription factor regulating cell cycle arrest, DNA repair, and apoptosis",
        "disease_associations": ["Li-Fraumeni syndrome", "Multiple cancers"],
        "critical_domains": ["DNA-binding domain (94-292)", "Tetramerization domain (323-356)"],
        "conservation": "Ultra-conserved: one of the most constrained genes in vertebrates"
    },
    "CFTR": {
        "full_name": "Cystic Fibrosis Transmembrane Conductance Regulator",
        "function": "Chloride and bicarbonate ion channel; regulates epithelial fluid transport",
        "disease_associations": ["Cystic fibrosis", "Congenital bilateral absence of vas deferens"],
        "critical_domains": ["Nucleotide-binding domain 1 (433-586)", "R domain (634-836)"],
        "conservation": "Highly conserved in vertebrates"
    },
    "EGFR": {
        "full_name": "Epidermal Growth Factor Receptor",
        "function": "Receptor tyrosine kinase; regulates cell proliferation and survival signaling",
        "disease_associations": ["Non-small cell lung cancer", "Glioblastoma"],
        "critical_domains": ["Ligand-binding domain (1-621)", "Kinase domain (712-979)"],
        "conservation": "Conserved across metazoans"
    },
    "KRAS": {
        "full_name": "Kirsten Rat Sarcoma Viral Proto-Oncogene",
        "function": "GTPase signal transducer; activates RAF-MEK-ERK proliferation pathway",
        "disease_associations": ["Pancreatic cancer", "Colorectal cancer", "Non-small cell lung cancer"],
        "critical_domains": ["GTPase domain (1-169)", "Hypervariable region (165-189)"],
        "conservation": "Ultra-conserved: identical across mammals at key codons"
    },
    "HBB": {
        "full_name": "Hemoglobin Subunit Beta",
        "function": "Oxygen transport in erythrocytes; forms hemoglobin tetramer with alpha subunits",
        "disease_associations": ["Sickle cell disease", "Beta-thalassemia"],
        "critical_domains": ["Heme-binding pocket", "Alpha-chain contact interface"],
        "conservation": "Highly conserved in vertebrates"
    },
    "PTEN": {
        "full_name": "Phosphatase and Tensin Homolog",
        "function": "Lipid phosphatase; negative regulator of PI3K-AKT survival pathway",
        "disease_associations": ["Cowden syndrome", "Prostate cancer", "Glioblastoma"],
        "critical_domains": ["Phosphatase domain (7-185)", "C2 domain (190-350)"],
        "conservation": "Highly conserved tumor suppressor"
    },
    "RB1": {
        "full_name": "Retinoblastoma Protein 1",
        "function": "Cell cycle regulator; blocks G1/S transition by sequestering E2F transcription factors",
        "disease_associations": ["Retinoblastoma", "Osteosarcoma"],
        "critical_domains": ["Pocket domain A (373-579)", "Pocket domain B (640-771)"],
        "conservation": "Conserved across vertebrates"
    },
}

# ─── Amino Acid Properties ───────────────────────────────────────────

AA_PROPERTIES = {
    'A': {'name': 'Alanine',       'type': 'nonpolar',   'charge': 'neutral',  'size': 'small'},
    'R': {'name': 'Arginine',      'type': 'polar',      'charge': 'positive', 'size': 'large'},
    'N': {'name': 'Asparagine',    'type': 'polar',      'charge': 'neutral',  'size': 'medium'},
    'D': {'name': 'Aspartate',     'type': 'polar',      'charge': 'negative', 'size': 'medium'},
    'C': {'name': 'Cysteine',      'type': 'polar',      'charge': 'neutral',  'size': 'small'},
    'Q': {'name': 'Glutamine',     'type': 'polar',      'charge': 'neutral',  'size': 'medium'},
    'E': {'name': 'Glutamate',     'type': 'polar',      'charge': 'negative', 'size': 'medium'},
    'G': {'name': 'Glycine',       'type': 'nonpolar',   'charge': 'neutral',  'size': 'tiny'},
    'H': {'name': 'Histidine',     'type': 'polar',      'charge': 'positive', 'size': 'medium'},
    'I': {'name': 'Isoleucine',    'type': 'nonpolar',   'charge': 'neutral',  'size': 'large'},
    'L': {'name': 'Leucine',       'type': 'nonpolar',   'charge': 'neutral',  'size': 'large'},
    'K': {'name': 'Lysine',        'type': 'polar',      'charge': 'positive', 'size': 'large'},
    'M': {'name': 'Methionine',    'type': 'nonpolar',   'charge': 'neutral',  'size': 'large'},
    'F': {'name': 'Phenylalanine', 'type': 'nonpolar',   'charge': 'neutral',  'size': 'large'},
    'P': {'name': 'Proline',       'type': 'nonpolar',   'charge': 'neutral',  'size': 'small'},
    'S': {'name': 'Serine',        'type': 'polar',      'charge': 'neutral',  'size': 'small'},
    'T': {'name': 'Threonine',     'type': 'polar',      'charge': 'neutral',  'size': 'medium'},
    'W': {'name': 'Tryptophan',    'type': 'nonpolar',   'charge': 'neutral',  'size': 'large'},
    'Y': {'name': 'Tyrosine',      'type': 'polar',      'charge': 'neutral',  'size': 'large'},
    'V': {'name': 'Valine',        'type': 'nonpolar',   'charge': 'neutral',  'size': 'medium'},
    '*': {'name': 'Stop codon',    'type': 'terminator', 'charge': 'none',     'size': 'none'},
}


# ─── Narrative Generation ────────────────────────────────────────────

def generate_biological_explanation(
    wt_protein: str,
    mut_protein: str,
    variant_annotation: str,
    mutation_position: int,
    mutation_type: str,
    hgvs_p: str = "",
    grantham_score: int = 0,
    embedding_distance: float = 0.0,
    structural_rmsd: float = 0.0,
    ddg: float = 0.0,
    active_site_proximity: bool = False,
    classifier_result: Optional[dict] = None,
    gene_name: Optional[str] = None,
) -> dict:
    """
    Generate a comprehensive biological explanation (Requirement #14).
    Produces a deterministic, template-based narrative paragraph.
    """
    codon_position = mutation_position // 3
    wt_aa = wt_protein[codon_position] if codon_position < len(wt_protein) else '?'
    mut_aa = mut_protein[codon_position] if codon_position < len(mut_protein) else '?'
    
    wt_props = AA_PROPERTIES.get(wt_aa, {'name': 'Unknown', 'type': 'neutral', 'charge': 'neutral', 'size': 'medium'})
    mut_props = AA_PROPERTIES.get(mut_aa, {'name': 'Unknown', 'type': 'neutral', 'charge': 'neutral', 'size': 'medium'})
    
    # Gene context
    gene_context = GENE_DATABASE.get(gene_name) if gene_name else None
    
    # Build molecular consequences
    molecular_consequences = _analyze_molecular_consequences(
        wt_aa, mut_aa, wt_props, mut_props, variant_annotation, codon_position
    )
    
    # Requirement #14: Narrative Report Generation (Deterministic Template)
    # The paragraph must state: type/pos, AA chemical consequence, 
    # structural impact (RMSD), ddG interpretation, active site risk, 
    # and overall verdict with confidence.
    
    # 1. Chemical Consequence
    chem_conseq = "no change"
    if wt_props['charge'] != mut_props['charge']:
        if mut_props['charge'] == 'positive': chem_conseq = "gain of positive charge"
        elif mut_props['charge'] == 'negative': chem_conseq = "gain of negative charge"
        else: chem_conseq = f"loss of {wt_props['charge']} charge"
    elif wt_props['type'] != mut_props['type']:
        if mut_props['type'] == 'nonpolar': chem_conseq = "gain of hydrophobicity"
        else: chem_conseq = f"change to {mut_props['type']} type"
        
    # 2. ddG Interpretation
    ddg_phrase = "neutral stability"
    if ddg > 1.0: ddg_phrase = "highly destabilizing"
    elif ddg > 0.3: ddg_phrase = "moderately destabilizing"
    elif ddg < -0.3: ddg_phrase = "stabilizing"
    
    # 3. Active site risk
    site_phrase = "within 8Å of a catalytic active site" if active_site_proximity else "not in immediate proximity to known active sites"
    
    # 4. Overall verdict
    verdict = classifier_result["classification"] if classifier_result else "VUS"
    conf = classifier_result["confidence"] * 100 if classifier_result else 0
    
    narrative = (
        f"This {mutation_type} mutation at position {mutation_position + 1} ({hgvs_p}) "
        f"results in a {wt_props['name']}-to-{mut_props['name']} substitution, causing a {chem_conseq}. "
        f"Structural analysis shows an impact of {structural_rmsd:.2f} Angstroms (RMSD), "
        f"which is interpreted as {ddg_phrase} (ΔΔG: {ddg:+.2f} kcal/mol). "
        f"The mutation site is {site_phrase}. "
        f"The overall simulation verdict is {verdict} with {conf:.1f}% confidence."
    )
    
    return {
        "summary": _generate_summary(wt_aa, mut_aa, wt_props, mut_props, variant_annotation, codon_position, gene_context, classifier_result),
        "detailed_narrative": narrative,
        "molecular_consequences": molecular_consequences,
        "confidence_level": "High" if conf > 75 else "Medium" if conf > 40 else "Low",
        "confidence_score": round(conf/100, 3)
    }

def _analyze_molecular_consequences(wt_aa, mut_aa, wt_props, mut_props, variant_annotation, position):
    consequences = []
    if variant_annotation == "Synonymous": consequences.append("Silent substitution - no residue change.")
    elif variant_annotation == "Nonsense": consequences.append("Premature termination.")
    elif variant_annotation == "Frameshift": consequences.append("Reading frame disrupted.")
    else:
        if wt_props['type'] != mut_props['type']: consequences.append(f"Polarity shift: {wt_props['type']} to {mut_props['type']}.")
        if wt_props['charge'] != mut_props['charge']: consequences.append(f"Charge altered: {wt_props['charge']} to {mut_props['charge']}.")
    return consequences

def _generate_summary(wt_aa, mut_aa, wt_props, mut_props, variant_annotation, position, gene_context, classifier_result):
    v = classifier_result["classification"] if classifier_result else "VUS"
    return f"{variant_annotation} variant ({wt_aa}{position+1}{mut_aa}). Predicted: {v}."

from app.models.schemas import MutationRequest, DetailedAnnotation
from typing import Tuple, Optional, List
import hashlib

def apply_mutation(sequence: str, mutation_type: str, position: int, new_nucleotide: str = None) -> str:
    """
    Applies a biological mutation to a given DNA sequence (Requirement #2).
    Validates that the specified position exists.
    """
    seq_list = list(sequence.upper())
    
    # Position validation (Requirement #2)
    if position < 0 or position >= len(seq_list) + (1 if mutation_type == "insertion" else 0):
        # Specific positioning rules for insertions (can occur at len) vs others
        if mutation_type == "insertion" and position == len(seq_list):
            pass
        else:
            raise ValueError(f"Mutation position {position} is out of bounds for sequence length {len(sequence)}.")
        
    new_nuc = new_nucleotide.upper() if new_nucleotide else None
    
    if mutation_type == "point":
        if not new_nuc or len(new_nuc) != 1:
            raise ValueError("Point mutation (substitution) requires exactly one new_nucleotide.")
        seq_list[position] = new_nuc
        
    elif mutation_type == "insertion":
        if not new_nuc:
            raise ValueError("Insertion requires a new_nucleotide sequence.")
        # Insert strings or chars
        for i, char in enumerate(new_nuc):
            seq_list.insert(position + i, char)
            
    elif mutation_type == "deletion":
        # position + 1 deletion? Requirement says starting at pos.
        if position >= len(seq_list):
             raise ValueError(f"Position {position} out of bounds for deletion.")
        seq_list.pop(position)
        
    else:
        raise ValueError(f"Unknown mutation_type: {mutation_type}")
        
    return "".join(seq_list)

def detect_frameshift(mutation_type: str, new_nucleotide: Optional[str]) -> bool:
    """Requirement #2: Detect if mutation causes a frameshift (length change not multiple of 3)."""
    if mutation_type == "point":
        return False
    
    change_len = 1 # default for point/del
    if mutation_type == "insertion" and new_nucleotide:
        change_len = len(new_nucleotide)
    elif mutation_type == "deletion":
        change_len = 1 # single base pop
        
    return change_len % 3 != 0

def annotate_variant(wt_aa_seq: str, mut_aa_seq: str, is_frameshift: bool) -> str:
    """
    Categorizes the mutation impact (Requirement #3).
    Labels: Synonymous, Missense, Nonsense, Frameshift.
    """
    if is_frameshift:
        return "Frameshift"
        
    # Same length check (silent/nonsense/missense)
    if len(wt_aa_seq) != len(mut_aa_seq):
        # In-frame indel
        return "In-frame Indel"
        
    differences = 0
    nonsense = False
    
    for wt_aa, mut_aa in zip(wt_aa_seq, mut_aa_seq):
        if wt_aa != mut_aa:
            differences += 1
            if mut_aa == '*' and wt_aa != '*':
                nonsense = True
                
    if differences == 0:
        return "Synonymous"
    elif nonsense:
        return "Nonsense"
    else:
        return "Missense"

# 3-letter AA mapping for HGVS (Requirement #4)
AA_3_LETTERS = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    '*': 'Ter', 'X': 'Xaa', '?': 'Unk'
}

def get_gnomad_conservation(wt_dna: str, position: int) -> float:
    """
    Requirement #10: Fallback to gnomAD allele frequency data.
    Simulated implementation: positions with low simulated frequency treated as conserved.
    Returns 0-1 (1 = highly conserved/low freq).
    """
    # Deterministic entropy based on position and sequence
    h = int(hashlib.md5(f"gnomad_{wt_dna}_{position}".encode()).hexdigest(), 16)
    simulated_allele_freq = (h % 10000) / 100000.0 # 0.0 to 0.1
    
    if simulated_allele_freq < 0.001:
        return 0.95 # Highly conserved
    return 1.0 - (simulated_allele_freq * 5) # gradient

def compute_sequence_fingerprint(sequence: str) -> List[int]:
    """
    Requirement #12: Compute deterministic radial waveform fingerprint.
    Samples at indices 0, 4, 8, 12, 16, 20, 24, 28 (wrapping).
    A=38, T=34, G=44, C=30. Add (char_code mod 12).
    """
    indices = [0, 4, 8, 12, 16, 20, 24, 28]
    base_radii = {'A': 38, 'T': 34, 'G': 44, 'C': 30, 'N': 32}
    
    fingerprint = []
    seq_len = len(sequence)
    if seq_len == 0:
        return [0] * 8
        
    for idx in indices:
        wrapped_idx = idx % seq_len
        base = sequence[wrapped_idx].upper()
        radius = base_radii.get(base, 32)
        # Add sequence-specific variation: char code mod 12
        variation = ord(base) % 12
        fingerprint.append(radius + variation)
        
    return fingerprint


def compute_detailed_annotation(
    wt_dna: str, 
    mut_dna: str, 
    wt_protein: str, 
    mut_protein: str, 
    mutation_type: str, 
    position: int, 
    new_nucleotide: Optional[str] = None
) -> DetailedAnnotation:
    """
    Requirement #2 & #4: Compute exhaustive variant metrics.
    """
    is_fs = detect_frameshift(mutation_type, new_nucleotide)
    variant_type = annotate_variant(wt_protein, mut_protein, is_fs)
    
    # HGVS c. notation (Requirement #4)
    orig_nuc = wt_dna[position] if position < len(wt_dna) else ""
    mut_nuc = new_nucleotide if new_nucleotide else ""
    hgvs_c = ""
    if mutation_type == "point":
        hgvs_c = f"c.{position + 1}{orig_nuc}>{mut_nuc}"
    elif mutation_type == "deletion":
        hgvs_c = f"c.{position + 1}del"
    elif mutation_type == "insertion":
        hgvs_c = f"c.{position + 1}_{position + 2}ins{mut_nuc}"
        
    codon_pos = position // 3
    wt_aa = wt_protein[codon_pos] if codon_pos < len(wt_protein) else "?"
    mut_aa = mut_protein[codon_pos] if codon_pos < len(mut_protein) else "?"
    
    # HGVS p. notation (Requirement #4)
    wt_aa_3 = AA_3_LETTERS.get(wt_aa, '?')
    mut_aa_3 = AA_3_LETTERS.get(mut_aa, '?')
    hgvs_p = f"p.{wt_aa_3}{codon_pos + 1}{mut_aa_3}"
    
    if variant_type == "Synonymous":
        hgvs_p = f"p.{wt_aa_3}{codon_pos + 1}="
    elif variant_type == "Nonsense":
        hgvs_p = f"p.{wt_aa_3}{codon_pos + 1}Ter"
    elif variant_type == "Frameshift":
        hgvs_p = f"p.{wt_aa_3}{codon_pos + 1}fs"
        
    codon_start = codon_pos * 3
    wt_codon = wt_dna[codon_start:codon_start+3]
    mut_codon = mut_dna[codon_start:codon_start+3]
    codon_change = f"{wt_codon} -> {mut_codon}" if len(wt_codon) == 3 and len(mut_codon) == 3 else ""
    
    amino_acid_change = f"{wt_aa_3} -> {mut_aa_3}"
    if variant_type == "Synonymous":
        amino_acid_change = "None"
        
    # Requirement #4: Functional domain lookup (simulated)
    domain_boundary = None
    if 100 <= codon_pos <= 200:
        domain_boundary = f"Kinase domain ({codon_pos})"
    
    # Requirement #10: Conservation lookup
    cs = round(get_gnomad_conservation(wt_dna, position), 3)
    
    # Requirement #2: Stop codon introduced
    stop_introduced = '*' in mut_protein and '*' not in wt_protein[:len(mut_protein)]
    
    return DetailedAnnotation(
        variant_type=variant_type,
        hgvs_c=hgvs_c,
        hgvs_p=hgvs_p,
        codon_change=codon_change,
        amino_acid_change=amino_acid_change,
        codon_affected=codon_pos + 1,
        stop_codon_introduced=stop_introduced,
        reading_frame_preserved=not is_fs,
        domain_boundary=domain_boundary,
        conservation_score=cs,
        high_risk_conservation=cs > 0.8
    )

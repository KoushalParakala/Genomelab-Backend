from app.models.schemas import QualityControlResult, TranslationResult

# Standard genetic code dictionary
CODON_TABLE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def perform_quality_control(sequence: str) -> QualityControlResult:
    """
    Run biological sequence QC checking for structural integrity (Requirement #1).
    Strips whitespace and line breaks silently.
    Validates characters (A, T, G, C, N) and length [10, 10000].
    """
    # Strip whitespace and line breaks (Requirement #1)
    sequence = "".join(sequence.split()).upper()
    errors = []
    warnings = []
    
    # 1. Length validation (Requirement #1)
    length = len(sequence)
    if length == 0:
        errors.append("Sequence is empty.")
    elif length < 10:
        errors.append(f"Sequence length ({length}) is below the minimum required 10 bp.")
    elif length > 10000:
        errors.append(f"Sequence length ({length}) exceeds the maximum supported 10,000 bp.")
        
    # 2. Check for invalid nucleotides (Requirement #1: A, T, G, C, N)
    valid_chars = set("ATGCN") 
    invalid_chars = set(sequence) - valid_chars
    if invalid_chars:
        errors.append(f"Invalid nucleotides detected: {', '.join(sorted(invalid_chars))}. Only A, T, G, C, N are permitted.")
        
    # Find first ATG for reading frame (Requirement #1)
    # The reading frame is typically defined by the position of the first start codon.
    start_codon_idx = sequence.find('ATG')
    reading_frame = start_codon_idx if start_codon_idx != -1 else 0
        
    # Generate overall valid state
    is_valid = len(errors) == 0
    
    # Calculate G/C Content
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = (gc_count / length * 100) if length > 0 else 0.0
    
    return QualityControlResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        length=length,
        gc_content=round(gc_content, 2),
        reading_frame=reading_frame
    )

def _transcribe(dna_sequence: str) -> str:
    """Convert DNA to mRNA (Requirement #3: Replace T with U)"""
    return dna_sequence.replace('T', 'U')

def simulate_translation(sequence: str) -> TranslationResult:
    """
    Transcribe to mRNA and translate to Amino Acid chain (Requirement #3).
    Translation begins at the first AUG and terminates at the first in-frame stop codon.
    """
    sequence = "".join(sequence.split()).upper()
    mrna = _transcribe(sequence)
    
    aa_chain = []
    stop_codon_detected = False
    
    # Find start codon (Requirement #3: translation begins at the first AUG)
    start_idx = mrna.find('AUG')
    if start_idx == -1:
        start_idx = 0  # fallback to frame 0 if no AUG
    
    # Only translate if there's enough sequence remaining
    remaining_len = len(mrna) - start_idx
    frame_shift = remaining_len % 3 != 0
    
    # Translate codon by codon (Requirement #3)
    for i in range(start_idx, start_idx + (remaining_len - (remaining_len % 3)), 3):
        # We need a bound check to avoid slicing past the end
        if i + 3 > len(mrna):
            break
            
        codon = mrna[i:i+3]
        if 'N' in codon: # Skip unknown codons
            aa_chain.append('X')
            continue
            
        aa = CODON_TABLE.get(codon, '?')
        
        if aa == '*':
            stop_codon_detected = True
            # We break upon reaching the first valid in-frame stop codon (Requirement #3)
            break
            
        aa_chain.append(aa)
    
    return TranslationResult(
        mrna_sequence=mrna,
        amino_acid_sequence="".join(aa_chain),
        stop_codon_detected=stop_codon_detected,
        frame_shift_detected=frame_shift
    )

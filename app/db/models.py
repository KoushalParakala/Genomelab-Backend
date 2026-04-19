from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
import uuid
from app.db.session import Base

def generate_uuid():
    return str(uuid.uuid4())

class SequenceRecord(Base):
    __tablename__ = "sequences"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    user_id = Column(String, nullable=True) # Optional link to profiles table
    sequence_data = Column(Text, nullable=False, index=True)
    length = Column(Integer, nullable=False)
    gc_content = Column(Float, nullable=False)
    is_valid = Column(Integer, default=1)  # 1 for Valid, 0 for Invalid
    translation = Column(Text, nullable=True) # Amino Acid representation
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class MutationLog(Base):
    __tablename__ = "mutation_logs"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    # Requirement #11 & #13: Session ID and Mutation Step for Replay Series
    session_id = Column(String, index=True, nullable=True)
    mutation_step = Column(Integer, default=0)
    
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=True)
    baseline_id = Column(String, ForeignKey("sequences.id"), nullable=True)
    
    original_sequence = Column(Text, nullable=True)
    mutated_sequence_data = Column(Text, nullable=False)
    mutation_type = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    new_nucleotide = Column(String, nullable=True)
    
    # Requirement #13: High-fidelity fields
    variant_annotation = Column(String, nullable=False) 
    hgvs_notation = Column(String, nullable=True) # HGVS DNA/Protein
    amino_acid_change = Column(String, nullable=True)
    verdict = Column(String, nullable=True) # Pathogenic, Benign, VUS
    
    # Requirement #13: AI Metrics
    stability_score = Column(Float, nullable=True) # Delta Delta G
    ddg = Column(Float, nullable=True) 
    structural_rmsd = Column(Float, nullable=True)
    plddt = Column(Float, nullable=True)
    pathogenicity_score = Column(Float, nullable=True)
    
    # Requirement #12 & #13: Sequence Fingerprint (8 values array)
    fingerprint = Column(JSON, nullable=True)
    
    # Requirement #13: PDB Structure (compressed blob / text)
    structure_pdb = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SharedResult(Base):
    __tablename__ = "shared_results"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    shared_by = Column(String, nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

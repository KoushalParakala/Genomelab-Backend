"""
Structure Prediction API Routes
=================================
Endpoints for protein structure prediction and comparison.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    StructurePredictionRequest, StructurePredictionResponse,
    StructureCompareRequest, StructureComparisonResult,
)

router = APIRouter()


@router.post("/predict", response_model=StructurePredictionResponse)
async def predict_structure_endpoint(request: StructurePredictionRequest):
    """
    Predict 3D protein structure from amino acid sequence using ESMFold.
    This is a synchronous call — may take 10-30s for longer sequences.
    """
    if len(request.protein_sequence) < 5:
        raise HTTPException(status_code=400, detail="Sequence must be at least 5 amino acids")
    
    if len(request.protein_sequence) > 400:
        raise HTTPException(status_code=400, detail="Sequence must be 400 residues or fewer for ESMFold API")
    
    from app.services.structure_service import predict_structure
    result = await predict_structure(request.protein_sequence)
    
    return StructurePredictionResponse(
        pdb_string=result["pdb_string"],
        sequence_length=result["sequence_length"],
        status=result["status"],
        error=result.get("error"),
    )


@router.post("/compare", response_model=StructureComparisonResult)
async def compare_structures_endpoint(request: StructureCompareRequest):
    """
    Compare two PDB structures by RMSD and per-residue displacement.
    """
    if not request.wt_pdb or not request.mut_pdb:
        raise HTTPException(status_code=400, detail="Both PDB strings are required")
    from app.services.structure_service import compare_structures
    # Note: request doesn't contain mutant positions, so we default for bare API usage
    result = compare_structures(request.wt_pdb, request.mut_pdb, "", "", 0)
    
    return StructureComparisonResult(**result)

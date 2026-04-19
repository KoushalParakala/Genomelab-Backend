from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, delete
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.api.auth_routes import get_current_user_id
from app.db.models import Experiment, MutationLog
from app.models.schemas import ExperimentCreate, ExperimentResponse

router = APIRouter()

@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    user_id: Optional[str] = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """List all experiments for the current user."""
    # Note: If Supabase connection via SQLAlchemy is used, this just works.
    # If not using Auth, user_id might be None, we return all or handle accordingly
    query = select(Experiment)
    if user_id:
        query = query.where(Experiment.user_id == user_id)
        
    result = await db.execute(query.order_by(Experiment.created_at.desc()))
    experiments = result.scalars().all()
    
    response = []
    for exp in experiments:
        # Get count (in a real app, do a group_by count query)
        count_query = select(MutationLog).where(MutationLog.experiment_id == exp.id)
        count_res = await db.execute(count_query)
        mutation_count = len(count_res.scalars().all())
        
        response.append(
            ExperimentResponse(
                id=exp.id,
                name=exp.name,
                description=exp.description,
                tags=exp.tags or [],
                created_at=str(exp.created_at) if exp.created_at else None,
                mutation_count=mutation_count
            )
        )
        
    return response

@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    user_id: Optional[str] = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Create a new experiment folder/tag for mutations."""
    if not user_id:
        # Fallback for local testing without auth
        user_id = "local-anon-user"
        
    from app.db.models import generate_uuid
    new_exp_id = generate_uuid()
    
    new_exp = Experiment(
        id=new_exp_id,
        user_id=user_id,
        name=experiment.name,
        description=experiment.description,
        tags=experiment.tags
    )
    db.add(new_exp)
    
    # Associate provided mutations
    for log_id in experiment.mutation_log_ids:
        query = select(MutationLog).where(MutationLog.id == log_id)
        res = await db.execute(query)
        mlog = res.scalar_one_or_none()
        if mlog:
            mlog.experiment_id = new_exp_id
            
    await db.commit()
    
    return ExperimentResponse(
        id=new_exp.id,
        name=new_exp.name,
        description=new_exp.description,
        tags=new_exp.tags or [],
        created_at=None,
        mutation_count=len(experiment.mutation_log_ids)
    )

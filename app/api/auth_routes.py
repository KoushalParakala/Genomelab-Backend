from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Optional
from app.core.config import settings

router = APIRouter()
security = HTTPBearer(auto_error=False)

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Validate Supabase JWT and return user ID. If not valid, return None."""
    if not credentials or not settings.SUPABASE_ANON_KEY:
        return None
    
    token = credentials.credentials
    try:
        # Supabase uses HS256 with the JWT secret (often same as anon key or configured explicitly)
        # For full verification, we'd need the JWT_SECRET.
        # As a fallback, we can use the Supabase client to get the user
        from app.db.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)
        if user and user.user:
            return user.user.id
        return None
    except Exception as e:
        return None

def require_current_user(user_id: Optional[str] = Depends(get_current_user_id)) -> str:
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid Supabase authentication token required",
        )
    return user_id

@router.get("/me")
async def get_current_profile(user_id: str = Depends(require_current_user)):
    """Get the current authenticated user's profile from Supabase."""
    from app.db.supabase_client import get_supabase_client
    supabase = get_supabase_client()
    response = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
    return response.data

import os
from supabase import create_client, Client
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Supabase client lazily
_supabase_client: Client | None = None
_supabase_admin: Client | None = None

def get_supabase_client() -> Client:
    """Get the standard Supabase client (Anon role)."""
    global _supabase_client
    if _supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
            logger.warning("Supabase URL or Anon Key not configured.")
            # Return a dummy or raise in a real strict env
            # For now we create it if available
        _supabase_client = create_client(
            settings.SUPABASE_URL or "https://placeholder.supabase.co",
            settings.SUPABASE_ANON_KEY or "placeholder_key"
        )
    return _supabase_client

def get_supabase_admin() -> Client:
    """Get the Supabase Admin client (Service Role) to bypass RLS if needed."""
    global _supabase_admin
    if _supabase_admin is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            logger.warning("Supabase URL or Service Key not configured.")
        _supabase_admin = create_client(
            settings.SUPABASE_URL or "https://placeholder.supabase.co",
            settings.SUPABASE_SERVICE_KEY or "placeholder_key"
        )
    return _supabase_admin

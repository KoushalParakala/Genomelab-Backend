from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from app.core.config import settings

# If USE_SUPABASE is true and SUPABASE_URL exists, construct postgres URL
# Assuming the user provides a direct postgres connection string via SUPABASE_DB_URL,
# OR we can fallback to SQLite. (We need SUPABASE_DB_URL for async SQLAlchemy)
DATABASE_URL = os.environ.get("SUPABASE_DB_URL", "sqlite+aiosqlite:///./dna_platform.db")

# Auto-fix Supabase connection strings to use the async driver
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=True, future=True)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with async_session() as session:
        yield session

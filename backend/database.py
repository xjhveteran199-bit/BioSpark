"""
Database connection and session management.

Supports SQLite (local dev) and PostgreSQL (production) via DATABASE_URL env var.
"""

import logging
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

log = logging.getLogger("biospark.db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./biospark.db",
)

# Railway provides postgres:// but SQLAlchemy needs postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Log driver + host (without password) on import so Railway logs surface
# whether we ended up on PostgreSQL (persistent) or SQLite (ephemeral on
# Railway's container filesystem — registrations get wiped on restart).
_driver = DATABASE_URL.split("://", 1)[0]
_after_at = DATABASE_URL.split("@", 1)[-1] if "@" in DATABASE_URL else DATABASE_URL.split("://", 1)[-1]
log.warning("DB engine initialized: driver=%s target=%s", _driver, _after_at)
if _driver.startswith("sqlite") and os.getenv("RAILWAY_ENVIRONMENT"):
    log.warning(
        "WARNING: SQLite on Railway is ephemeral — user accounts will be lost "
        "on every redeploy/restart. Attach a PostgreSQL plugin and set DATABASE_URL."
    )

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables. Called on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.warning("DB tables created/verified")


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        yield session

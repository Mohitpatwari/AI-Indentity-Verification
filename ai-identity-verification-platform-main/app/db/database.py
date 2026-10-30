"""Database engine and session setup for SQLite persistence."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./verification_history.db"

# check_same_thread=False is needed because SQLite by default only allows
# one thread to use a connection — FastAPI/uvicorn handles requests across
# multiple threads, so we relax this restriction (safe for our use case
# since SQLAlchemy manages a session per request, not shared connections).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency that provides a database session per request, and
    guarantees it's closed afterward even if an error occurs."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""
storage/db.py

SQLite persistence layer via SQLAlchemy.
Stores all structured data (cards, meta decks, matchups, tier lists)
and the scrape_log cache table that prevents redundant fetches.
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, inspect
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///knowledge.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Tables ──────────────────────────────────────────────────────────────────

class Card(Base):
    """All Clash Royale cards with base stats."""
    __tablename__ = "cards"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    elixir_cost = Column(Integer)
    card_type = Column(String)   # Troop / Spell / Building
    rarity = Column(String)      # Common / Rare / Epic / Legendary / Champion
    description = Column(Text)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MetaDeck(Base):
    """Top ladder deck compositions with usage and win rates."""
    __tablename__ = "meta_decks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deck_hash = Column(String, unique=True, nullable=False)  # sorted card names joined
    cards_json = Column(Text, nullable=False)                # JSON array of card names
    usage_rate = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    season = Column(String)
    source = Column(String)                                  # "royaleapi" | "deckshop"
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Matchup(Base):
    """Deck-vs-deck win probabilities."""
    __tablename__ = "matchups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deck_a_hash = Column(String, nullable=False)
    deck_b_hash = Column(String, nullable=False)
    win_probability = Column(Float)   # probability deck_a beats deck_b
    sample_size = Column(Integer, default=0)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TierList(Base):
    """Card tier classifications from Deckshop / community sources."""
    __tablename__ = "tier_lists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_name = Column(String, nullable=False)
    tier = Column(String)       # S / A / B / C / D
    source = Column(String)     # "deckshop" | "community"
    meta_label = Column(String) # e.g. "2.6 cycle meta"
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ScrapeLog(Base):
    """
    Cache control table.
    Every ingestion module checks this before making any network request.
    If last_scraped is within ttl_hours, the scrape is skipped.
    """
    __tablename__ = "scrape_log"

    source = Column(String, primary_key=True)   # e.g. "royale_api.cards"
    last_scraped = Column(DateTime)
    status = Column(String, default="ok")        # "ok" | "error"
    record_count = Column(Integer, default=0)
    ttl_hours = Column(Integer, default=24)
    error_message = Column(Text, nullable=True)


# ── Schema Init ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    print("[db] Schema initialised.")


# ── Cache Helpers ─────────────────────────────────────────────────────────────

def get_scrape_log(source: str) -> ScrapeLog | None:
    with SessionLocal() as session:
        return session.get(ScrapeLog, source)


def should_scrape(source: str) -> bool:
    """
    Returns True if data for this source is missing or stale.
    Returns False if data is fresh (within TTL) — skip the scrape.
    """
    log = get_scrape_log(source)
    if log is None or log.last_scraped is None:
        return True
    if log.status == "error":
        return True  # always retry after an error
    age_hours = (datetime.now(timezone.utc) - log.last_scraped.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    return age_hours >= log.ttl_hours


def mark_scraped(
    source: str,
    record_count: int = 0,
    ttl_hours: int = 24,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    """Update or insert a scrape_log entry after a run."""
    with SessionLocal() as session:
        log = session.get(ScrapeLog, source)
        if log is None:
            log = ScrapeLog(source=source)
            session.add(log)
        log.last_scraped = datetime.now(timezone.utc)
        log.status = status
        log.record_count = record_count
        log.ttl_hours = ttl_hours
        log.error_message = error_message
        session.commit()
    print(f"[db] scrape_log updated: {source} | status={status} | records={record_count}")


# ── Generic Upsert ────────────────────────────────────────────────────────────

def upsert_cards(cards: list[dict]) -> int:
    """Insert or update cards by name."""
    with SessionLocal() as session:
        count = 0
        for c in cards:
            existing = session.query(Card).filter_by(name=c["name"]).first()
            if existing:
                for k, v in c.items():
                    setattr(existing, k, v)
            else:
                session.add(Card(**c))
                count += 1
        session.commit()
    return count


def upsert_meta_decks(decks: list[dict]) -> int:
    """Insert or update meta decks by deck_hash."""
    with SessionLocal() as session:
        count = 0
        for d in decks:
            existing = session.query(MetaDeck).filter_by(deck_hash=d["deck_hash"]).first()
            if existing:
                for k, v in d.items():
                    setattr(existing, k, v)
            else:
                session.add(MetaDeck(**d))
                count += 1
        session.commit()
    return count


def upsert_matchups(matchups: list[dict]) -> int:
    """Insert or skip matchups (deck_a + deck_b combo)."""
    with SessionLocal() as session:
        count = 0
        for m in matchups:
            existing = (
                session.query(Matchup)
                .filter_by(deck_a_hash=m["deck_a_hash"], deck_b_hash=m["deck_b_hash"])
                .first()
            )
            if existing:
                existing.win_probability = m["win_probability"]
                existing.sample_size = m.get("sample_size", 0)
            else:
                session.add(Matchup(**m))
                count += 1
        session.commit()
    return count


def upsert_tier_list(entries: list[dict]) -> int:
    """Insert new tier list entries."""
    with SessionLocal() as session:
        for e in entries:
            session.add(TierList(**e))
        session.commit()
    return len(entries)


if __name__ == "__main__":
    init_db()

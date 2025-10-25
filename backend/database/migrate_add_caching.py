"""Database migration to add CachedMarketData table"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from database.connection import engine
from database.models import Base, CachedMarketData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add CachedMarketData table to existing database"""
    try:
        logger.info("Starting migration: Adding CachedMarketData table...")
        
        # Create only the CachedMarketData table
        CachedMarketData.__table__.create(bind=engine, checkfirst=True)
        
        logger.info("✅ Migration completed successfully!")
        logger.info("✅ CachedMarketData table created (if it didn't exist)")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    migrate()



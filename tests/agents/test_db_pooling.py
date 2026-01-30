from unittest.mock import patch, MagicMock
from sqlalchemy.pool import NullPool
from kavalai.agents.db import DatabaseManager


def test_database_manager_pooling():
    db_manager = DatabaseManager()

    # Test NullPool (default)
    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        db_manager.get_sessionmaker(uri="postgresql://localhost/test")
        mock_create_engine.assert_called_with(
            "postgresql+asyncpg://localhost/test", echo=False, poolclass=NullPool
        )

    # Reset engines for next test
    db_manager._engines = {}

    # Test QueuePool (pool_size > 0)
    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        db_manager.get_sessionmaker(
            uri="postgresql://localhost/test", pool_size=5, max_overflow=10
        )
        mock_create_engine.assert_called_with(
            "postgresql+asyncpg://localhost/test",
            echo=False,
            pool_size=5,
            max_overflow=10,
        )


def test_database_manager_caching():
    db_manager = DatabaseManager()
    uri = "postgresql://localhost/test"

    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # First call creates engine
        db_manager.get_sessionmaker(uri=uri)
        assert mock_create_engine.call_count == 1

        # Second call uses cached engine
        db_manager.get_sessionmaker(uri=uri)
        assert mock_create_engine.call_count == 1

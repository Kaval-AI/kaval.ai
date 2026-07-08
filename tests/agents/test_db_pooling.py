import pytest
from unittest.mock import patch, MagicMock
from kavalai.agents.db import DatabaseManager, EngineOptionsConflictError


def test_database_manager_pooling():
    db_manager = DatabaseManager()

    # Test default (pool_size=1)
    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        db_manager.get_sessionmaker(uri="postgresql://localhost/test")
        mock_create_engine.assert_called_with(
            "postgresql+asyncpg://localhost/test",
            echo=False,
            pool_size=1,
            max_overflow=0,
        )

    # Reset engines for next test
    db_manager._engines = {}
    db_manager._engine_options = {}

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


def test_cache_hit_with_matching_explicit_options():
    """Re-requesting the engine's effective options is not a conflict."""
    db_manager = DatabaseManager()
    uri = "postgresql://localhost/test"

    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        db_manager.get_sessionmaker(uri=uri, pool_size=5, max_overflow=10)

        # Same explicit options, a subset, and fully unspecified all reuse it.
        db_manager.get_sessionmaker(uri=uri, pool_size=5, max_overflow=10)
        db_manager.get_sessionmaker(uri=uri, pool_size=5)
        db_manager.get_sessionmaker(uri=uri)
        assert mock_create_engine.call_count == 1


def test_cache_hit_with_conflicting_options_raises():
    """Explicitly requesting different options than the cached engine raises."""
    db_manager = DatabaseManager()
    uri = "postgresql://user:secret@localhost/test"

    with patch("kavalai.agents.db.create_async_engine"):
        db_manager.get_sessionmaker(uri=uri, pool_size=1, max_overflow=0)

        with pytest.raises(EngineOptionsConflictError) as exc_info:
            db_manager.get_sessionmaker(uri=uri, pool_size=20)

        message = str(exc_info.value)
        assert "pool_size" in message
        assert "20" in message
        # The message must not leak the database password.
        assert "secret" not in message

        # echo conflicts are detected too.
        with pytest.raises(EngineOptionsConflictError):
            db_manager.get_sessionmaker(uri=uri, echo=True)


def test_conflict_against_default_created_engine():
    """An engine created with defaults conflicts with later explicit options."""
    db_manager = DatabaseManager()
    uri = "postgresql://localhost/test"

    with patch("kavalai.agents.db.create_async_engine"):
        db_manager.get_sessionmaker(uri=uri)  # effective: echo=False, 1/0

        # Explicitly matching the applied defaults is fine.
        db_manager.get_sessionmaker(uri=uri, pool_size=1, max_overflow=0, echo=False)

        with pytest.raises(EngineOptionsConflictError):
            db_manager.get_sessionmaker(uri=uri, max_overflow=10)


def test_conflict_is_scoped_per_url_and_schema():
    """Different (url, schema) keys have independent engines and options."""
    db_manager = DatabaseManager()
    uri = "postgresql://localhost/test"

    with patch("kavalai.agents.db.create_async_engine") as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        db_manager.get_sessionmaker(uri=uri, schema="tenant_a", pool_size=1)

        # A different schema is a fresh engine; different options are allowed.
        db_manager.get_sessionmaker(uri=uri, schema="tenant_b", pool_size=20)
        assert mock_create_engine.call_count == 2

        # But each keeps enforcing its own creation options.
        with pytest.raises(EngineOptionsConflictError):
            db_manager.get_sessionmaker(uri=uri, schema="tenant_a", pool_size=20)
        with pytest.raises(EngineOptionsConflictError):
            db_manager.get_sessionmaker(uri=uri, schema="tenant_b", pool_size=1)

import csv
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from kavalai.tools.index_csv import index_csv, main
from kavalai.agents.db import EmbeddingProfile


@pytest.fixture
def temp_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    content = [
        {"id": "1", "name": "Alice", "secret": "shh", "bio": "Hello Alice"},
        {"id": "2", "name": "Bob", "secret": "secret", "bio": "Hi Bob"},
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "secret", "bio"])
        writer.writeheader()
        writer.writerows(content)
    return str(csv_file)


@pytest.fixture
def multi_row_csv(tmp_path):
    csv_file = tmp_path / "multi_row.csv"
    content = [
        {"id": "1", "name": "Alice", "secret": "shh", "bio": "Line 1\nLine 2"},
        {"id": "2", "name": "Bob", "secret": "secret", "bio": "Line 3"},
        {"id": "3", "name": "Charlie", "secret": "secret", "bio": "Line 4\nLine 5"},
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "secret", "bio"])
        writer.writeheader()
        writer.writerows(content)
    return str(csv_file)


@pytest.mark.asyncio
async def test_index_csv_index_fields(temp_csv):
    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=["id"],
            index_fields=["bio"],
            mode="full",
            limit=None,
            batch_size=1,
        )

        # Check first call content
        # Only "bio" should be in texts
        call_args = mock_rag_service.batch_index.call_args_list[0]
        texts = call_args[0][0]
        metas = call_args[0][1]

        assert texts[0] == "Hello Alice"
        assert "name:" not in texts[0]
        assert "id:" not in texts[0]
        assert metas[0] == {"id": "1"}


import os


@pytest.mark.asyncio
async def test_index_csv_success(temp_csv):
    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        # Mock Session
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        # Mock AgentService
        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        # Mock RagService
        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=["id", "name"],
            index_fields=["bio"],
            mode="full",
            limit=None,
            batch_size=2,
        )

        # Verify RagService.batch_index was called once (batch_size=2, 2 rows)
        assert mock_rag_service.batch_index.call_count == 1

        # Check first call content
        # Content: "Hello Alice" (only bio is indexed)
        # Meta: {"id": "1", "name": "Alice"}
        call_args = mock_rag_service.batch_index.call_args_list[0]
        texts, metas, collection_name = (
            call_args[0][0],
            call_args[0][1],
            call_args[1]["collection_name"],
        )
        assert texts[0] == "Hello Alice"
        assert metas[0] == {"id": "1", "name": "Alice"}
        assert collection_name == "test_collection"


@pytest.mark.asyncio
async def test_index_csv_limit(temp_csv):
    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=[],
            index_fields=["bio"],
            mode="full",
            limit=1,
            batch_size=10,
        )

        # Verify only 1 row was indexed
        assert mock_rag_service.batch_index.call_count == 1
        texts = mock_rag_service.batch_index.call_args[0][0]
        assert len(texts) == 1


@pytest.mark.asyncio
async def test_index_csv_split_mode_lines(temp_csv):
    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=[],
            index_fields=["bio"],
            mode="lines",
            limit=1,
            batch_size=1,
        )

        texts = mock_rag_service.batch_index.call_args[0][0]
        assert "Hello Alice" in texts[0]


def test_main_arg_parsing(temp_csv):
    with patch(
        "sys.argv",
        [
            "index_csv.py",
            temp_csv,
            "--collection-name",
            "test",
            "--embedding-profile",
            "test-prof",
            "--index-fields",
            "bio",
        ],
    ), patch("kavalai.tools.index_csv.asyncio.run") as mock_run:
        main()
        assert mock_run.called


def test_main_file_not_found():
    with patch(
        "sys.argv",
        [
            "index_csv.py",
            "non_existent.csv",
            "--collection-name",
            "test",
            "--embedding-profile",
            "test-prof",
            "--index-fields",
            "bio",
        ],
    ), patch("sys.exit") as mock_exit:
        main()
        assert mock_exit.called


@pytest.mark.asyncio
async def test_index_csv_batch_by_rows(multi_row_csv):
    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        # CSV has 3 rows. Batch size 2.
        # Batch 1: Rows 1 and 2.
        # Row 1 has 2 lines, Row 2 has 1 line. Total 3 chunks.
        # Batch 2: Row 3.
        # Row 3 has 2 lines. Total 2 chunks.
        await index_csv(
            csv_path=multi_row_csv,
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=["id"],
            index_fields=["bio"],
            mode="lines",
            limit=None,
            batch_size=2,
        )

        assert mock_rag_service.batch_index.call_count == 2

        # Check first call
        first_call_texts = mock_rag_service.batch_index.call_args_list[0][0][0]
        assert (
            len(first_call_texts) == 3
        )  # Line 1, Line 2 (from row 1), Line 3 (from row 2)
        assert "Line 1" in first_call_texts
        assert "Line 2" in first_call_texts
        assert "Line 3" in first_call_texts

        # Check second call
        second_call_texts = mock_rag_service.batch_index.call_args_list[1][0][0]
        assert len(second_call_texts) == 2  # Line 4, Line 5 (from row 3)
        assert "Line 4" in second_call_texts
        assert "Line 5" in second_call_texts


@pytest.mark.asyncio
async def test_index_csv_profile_not_found(temp_csv):
    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path", return_value=None
    ):
        # Should return early without erroring out on DB stuff
        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection",
            embedding_profile_name="non-existent",
            metadata_fields=[],
            index_fields=["bio"],
            mode="full",
            limit=None,
            batch_size=1,
        )


@pytest.mark.asyncio
async def test_index_csv_lines_split(tmp_path):
    csv_file = tmp_path / "split_test.csv"
    content = [
        {"id": "1", "bio": "Line 1\nLine 2\n\nLine 3"},
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "bio"])
        writer.writeheader()
        writer.writerows(content)

    mock_profile = EmbeddingProfile(
        id="00000000-0000-0000-0000-000000000000",
        name="test-profile",
        provider="openai",
        model_name="text-embedding-3-small",
    )

    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=mock_profile,
    ), patch("kavalai.tools.index_csv.db_manager") as mock_db_manager, patch(
        "kavalai.tools.index_csv.AgentService"
    ) as mock_agent_service_cls, patch(
        "kavalai.tools.index_csv.RagService"
    ) as mock_rag_service_cls, patch.dict(
        os.environ,
        {
            "AGENTS_DB_USER": "user",
            "AGENTS_DB_PASSWORD": "pass",
            "AGENTS_DB_HOST": "host",
            "AGENTS_DB_PORT": "5432",
            "AGENTS_DB_NAME": "db",
        },
    ):
        mock_session = AsyncMock()
        mock_db_manager.get_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session

        mock_agent_service = MagicMock()
        mock_agent_service.upsert_embedding_profile = AsyncMock(
            return_value=mock_profile
        )
        mock_agent_service_cls.return_value = mock_agent_service

        mock_rag_service = MagicMock()
        mock_rag_service.batch_index = AsyncMock()
        mock_rag_service_cls.return_value = mock_rag_service

        await index_csv(
            csv_path=str(csv_file),
            collection_name="test_collection",
            embedding_profile_name="test-profile",
            metadata_fields=["id"],
            index_fields=["bio"],
            mode="lines",
            limit=None,
            batch_size=10,
        )

        # Should be called once with 3 texts
        assert mock_rag_service.batch_index.call_count == 1
        call_args = mock_rag_service.batch_index.call_args
        texts = call_args[0][0]
        metas = call_args[0][1]

        assert len(texts) == 3
        assert texts == ["Line 1", "Line 2", "Line 3"]
        assert all(m == {"id": "1"} for m in metas)

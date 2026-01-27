import csv
import pytest
import os
from sqlalchemy import select
from kavalai.tools.index_csv import index_csv, main
from kavalai.agents.db import EmbeddingProfile, RagIndex
from unittest.mock import patch


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


@pytest.fixture
def test_profile_obj():
    return EmbeddingProfile(
        name="test-openai",
        provider="openai",
        model_name="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY", "sk-test"),
    )


@pytest.fixture(autouse=True)
def mock_index_csv_deps(test_profile_obj, agents_session_maker):
    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path",
        return_value=test_profile_obj,
    ), patch(
        "kavalai.tools.index_csv.db_manager.get_sessionmaker",
        return_value=agents_session_maker,
    ):
        yield


@pytest.mark.asyncio
async def test_index_csv_index_fields(temp_csv, agents_db, test_profile_obj):
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=["id"],
        index_fields=["bio"],
        source_field="id",
        mode="full",
        limit=None,
        batch_size=1,
    )

    # Verify in DB
    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection")
    )
    items = result.scalars().all()
    assert len(items) == 2

    # Sort by source_id for easier checking
    items = sorted(items, key=lambda x: x.source_id)

    assert items[0].content == "Hello Alice"
    assert items[0].rag_metadata == {"id": "1"}
    assert items[0].source_id == "1"
    # The profile will be upserted in index_csv, so we check its name or id
    # We can retrieve the upserted profile from DB
    assert len(items[0].embedding) == 1536


@pytest.mark.asyncio
async def test_index_csv_success(temp_csv, agents_db, test_profile_obj):
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection_success",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=["id", "name"],
        index_fields=["bio"],
        source_field="id",
        mode="full",
        limit=None,
        batch_size=2,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_success")
    )
    items = result.scalars().all()
    assert len(items) == 2

    items = sorted(items, key=lambda x: x.source_id)
    assert items[0].content == "Hello Alice"
    assert items[0].rag_metadata == {"id": "1", "name": "Alice"}
    assert items[1].source_id == "2"


@pytest.mark.asyncio
async def test_index_csv_limit(temp_csv, agents_db, test_profile_obj):
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection_limit",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=[],
        index_fields=["bio"],
        source_field="id",
        mode="full",
        limit=1,
        batch_size=10,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_limit")
    )
    items = result.scalars().all()
    assert len(items) == 1
    assert items[0].content == "Hello Alice"


@pytest.mark.asyncio
async def test_index_csv_split_mode_lines(temp_csv, agents_db, test_profile_obj):
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection_lines",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=[],
        index_fields=["bio"],
        source_field="id",
        mode="lines",
        limit=1,
        batch_size=1,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_lines")
    )
    items = result.scalars().all()
    assert len(items) == 1
    assert "Hello Alice" in items[0].content


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
            "--source-field",
            "id",
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
            "--source-field",
            "id",
        ],
    ), pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1


@pytest.mark.asyncio
async def test_index_csv_batch_by_rows(multi_row_csv, agents_db, test_profile_obj):
    # CSV has 3 rows. Batch size 2.
    # Batch 1: Rows 1 and 2.
    # Row 1 has 2 lines, Row 2 has 1 line. Total 3 chunks.
    # Batch 2: Row 3.
    # Row 3 has 2 lines. Total 2 chunks.
    await index_csv(
        csv_path=multi_row_csv,
        collection_name="test_collection_batch",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=["id"],
        index_fields=["bio"],
        source_field="id",
        mode="lines",
        limit=None,
        batch_size=2,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_batch")
    )
    items = result.scalars().all()
    assert len(items) == 5

    contents = [item.content for item in items]
    assert "Line 1" in contents
    assert "Line 2" in contents
    assert "Line 3" in contents
    assert "Line 4" in contents
    assert "Line 5" in contents


@pytest.mark.asyncio
async def test_index_csv_profile_not_found(temp_csv, agents_db):
    # We need to override the autouse mock for this specific test
    with patch(
        "kavalai.tools.index_csv.load_embedding_profile_from_path", return_value=None
    ):
        await index_csv(
            csv_path=temp_csv,
            collection_name="test_collection_fail",
            embedding_profile_name="non-existent",
            metadata_fields=[],
            index_fields=["bio"],
            source_field="id",
            mode="full",
            limit=None,
            batch_size=1,
        )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_fail")
    )
    items = result.scalars().all()
    assert len(items) == 0


@pytest.mark.asyncio
async def test_index_csv_lines_split(tmp_path, agents_db, test_profile_obj):
    csv_file = tmp_path / "split_test.csv"
    content = [
        {"id": "1", "bio": "Line 1\nLine 2\n\nLine 3"},
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "bio"])
        writer.writeheader()
        writer.writerows(content)

    await index_csv(
        csv_path=str(csv_file),
        collection_name="test_collection_split",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=["id"],
        index_fields=["bio"],
        source_field="id",
        mode="lines",
        limit=None,
        batch_size=10,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_split")
    )
    items = result.scalars().all()
    assert len(items) == 3
    contents = sorted([item.content for item in items])
    assert contents == ["Line 1", "Line 2", "Line 3"]
    assert all(item.rag_metadata == {"id": "1"} for item in items)


@pytest.mark.asyncio
async def test_index_csv_replace(temp_csv, agents_db, test_profile_obj):
    # First index
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection_replace",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=[],
        index_fields=["bio"],
        source_field="id",
        mode="full",
        limit=None,
        batch_size=10,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_replace")
    )
    assert len(result.scalars().all()) == 2

    # Index again with replace=True
    await index_csv(
        csv_path=temp_csv,
        collection_name="test_collection_replace",
        embedding_profile_name=test_profile_obj.name,
        metadata_fields=[],
        index_fields=["bio"],
        source_field="id",
        mode="full",
        limit=None,
        replace=True,
        batch_size=10,
    )

    result = await agents_db.execute(
        select(RagIndex).where(RagIndex.collection_name == "test_collection_replace")
    )
    items = result.scalars().all()
    # If replace works, it should still be 2, not 4
    assert len(items) == 2

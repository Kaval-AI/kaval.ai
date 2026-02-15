import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from kavalai.agents.rag_service import RagService
from kavalai.normalizer import Normalizer
from kavalai.agents.db import ModelCallStat


@pytest.mark.asyncio
async def test_rag_service_with_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    # Mock session maker
    mock_session_maker = MagicMock()
    mock_session_maker.return_value.__aenter__.return_value = mock_session

    model = "openai/text-embedding-3-small"
    normalizer = Normalizer(l2=True)

    # We need to mock compute_embeddings in rag_service
    with patch("kavalai.agents.rag_service.compute_embeddings") as mock_compute:
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_compute.return_value = ([[0.1, 0.2, 0.3]], mock_stats)

        # Initialize RagService with normalizer
        # Pass a session instead of URI to avoid real DB engine creation
        service = RagService(
            uri_or_session=mock_session, model=model, normalizer=normalizer
        )

        assert service.normalizer == normalizer

        # 1. Test batch_index
        await service.batch_index(
            texts=["test"], metadata_list=[{}], collection_name="test_coll"
        )

        mock_compute.assert_called_with(
            model=model, texts=["test"], normalizer=normalizer
        )

        # 2. Test query
        mock_compute.reset_mock()
        await service.query("test query")

        mock_compute.assert_called_with(
            model=model, texts=["test query"], normalizer=normalizer
        )

        # 3. Test compute_similarity_matrix
        mock_compute.reset_mock()
        await service.compute_similarity_matrix(texts=["t1"], source_ids=["s1"])

        mock_compute.assert_called_with(
            model=model, texts=["t1"], normalizer=normalizer
        )


@pytest.mark.asyncio
async def test_rag_service_without_normalizer():
    # Mock database session
    mock_session = AsyncMock()
    mock_session.execute.return_value = MagicMock(all=MagicMock(return_value=[]))

    model = "openai/text-embedding-3-small"

    with patch("kavalai.agents.rag_service.compute_embeddings") as mock_compute:
        mock_stats = MagicMock(spec=ModelCallStat)
        mock_compute.return_value = ([[0.1, 0.2, 0.3]], mock_stats)

        # Initialize RagService without normalizer
        service = RagService(uri_or_session=mock_session, model=model)

        assert service.normalizer is None

        await service.query("test query")

        mock_compute.assert_called_with(
            model=model, texts=["test query"], normalizer=None
        )

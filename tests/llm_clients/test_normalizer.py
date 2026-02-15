import os
import pytest
import math
import numpy as np
from kavalai.normalizer import Normalizer
from kavalai.agents.db import RagIndex
from kavalai.agents.rag_service import RagService


def test_normalize_l1():
    embeddings = [[1.0, -1.0], [0.0, 0.0], [3.0, 4.0]]
    normalizer = Normalizer()
    # Manual call to normalize_l1 now expects numpy array and returns numpy array
    normalized = normalizer.normalize_l1(np.array(embeddings)).tolist()

    # [1.0, -1.0] -> sum(abs) = 2.0 -> [0.5, -0.5]
    assert normalized[0] == [0.5, -0.5]
    assert sum(abs(x) for x in normalized[0]) == 1.0

    # [0.0, 0.0] -> sum(abs) = 0 -> [0.0, 0.0]
    assert normalized[1] == [0.0, 0.0]

    # [3.0, 4.0] -> sum(abs) = 7.0 -> [3/7, 4/7]
    assert pytest.approx(normalized[2]) == [3 / 7, 4 / 7]
    assert pytest.approx(sum(abs(x) for x in normalized[2])) == 1.0


def test_normalize_l2():
    embeddings = [[1.0, 1.0], [0.0, 0.0], [3.0, 4.0]]
    normalizer = Normalizer()
    normalized = normalizer.normalize_l2(np.array(embeddings)).tolist()

    # [1.0, 1.0] -> norm is sqrt(2), so [1/sqrt(2), 1/sqrt(2)]
    assert pytest.approx(sum(x * x for x in normalized[0])) == 1.0

    # [0.0, 0.0] -> norm is 0, should remain [0.0, 0.0]
    assert normalized[1] == [0.0, 0.0]

    # [3.0, 4.0] -> norm is 5, so [0.6, 0.8]
    assert normalized[2] == [0.6, 0.8]
    assert pytest.approx(sum(x * x for x in normalized[2])) == 1.0


def test_centering():
    center_vector = [1.0, 2.0]
    embeddings = [[2.0, 3.0], [1.0, 2.0], [0.0, 0.0]]
    normalizer = Normalizer(center_vector=center_vector)
    centered = normalizer.center(np.array(embeddings)).tolist()

    assert centered[0] == [1.0, 1.0]
    assert centered[1] == [0.0, 0.0]
    assert centered[2] == [-1.0, -2.0]


def test_transform():
    center_vector = [1.0, 1.0]
    embeddings = [[2.0, 2.0]]
    normalizer = Normalizer(center_vector=center_vector, l2=True, center=True)

    # Center: [2, 2] - [1, 1] = [1, 1]
    # L2: [1, 1] -> [1/sqrt(2), 1/sqrt(2)]
    result = normalizer.transform(embeddings)
    assert pytest.approx(result[0]) == [1 / math.sqrt(2), 1 / math.sqrt(2)]


def test_transform_single_vector():
    center_vector = [1.0, 1.0]
    embedding = [2.0, 2.0]
    normalizer = Normalizer(center_vector=center_vector, l2=True, center=True)

    result = normalizer.transform(embedding)
    assert pytest.approx(result) == [1 / math.sqrt(2), 1 / math.sqrt(2)]


def test_yaml_string_load_save():
    center_vector = [1.0, 2.0, 3.0]
    original = Normalizer(center_vector=center_vector, l1=True, l2=False, center=True)
    yaml_str = original.to_yaml()

    assert "l1: true" in yaml_str.lower()
    assert "center: true" in yaml_str.lower()

    loaded = Normalizer.from_yaml(yaml_str)
    assert loaded.l1 is True
    assert loaded.l2 is False
    assert loaded.center_enabled is True
    assert np.allclose(loaded.center_vector, center_vector)


def test_yaml_save_load(tmp_path):
    yaml_path = os.path.join(tmp_path, "normalizer.yaml")
    center_vector = [1.0, 2.0, 3.0]
    original = Normalizer(center_vector=center_vector, l1=True, l2=False, center=True)
    original.save_to_yaml(yaml_path)

    loaded = Normalizer.load_from_yaml(yaml_path)
    assert loaded.l1 is True
    assert loaded.l2 is False
    assert loaded.center_enabled is True
    assert np.allclose(loaded.center_vector, center_vector)

    # Verify it works
    test_emb = [[2.0, 3.0, 4.0]]
    # Center: [1, 1, 1]
    # L1: [1/3, 1/3, 1/3]
    result = loaded.transform(test_emb)
    assert pytest.approx(result[0]) == [1 / 3, 1 / 3, 1 / 3]


@pytest.mark.asyncio
async def test_learn_from_rag(agents_db):
    # Add some data to RagIndex
    model = "openai/test-model"
    collection = "test-collection"

    embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    for i, emb in enumerate(embeddings):
        idx = RagIndex(
            model=model,
            collection_name=collection,
            source_id=f"src_{i}",
            embedding=emb,
            embedding_size=2,
        )
        agents_db.add(idx)

    await agents_db.commit()

    # Learn from RAG
    normalizer = await Normalizer.learn_from_rag(
        agents_db, model=model, collection_name=collection
    )

    # Mean should be [ (1+3+5)/3, (2+4+6)/3 ] = [3.0, 4.0]
    assert np.allclose(normalizer.center_vector, [3.0, 4.0])

    # Test centering with learned vector
    test_emb = [[3.0, 4.0], [0.0, 0.0]]
    centered = normalizer.center(np.array(test_emb)).tolist()
    assert centered[0] == [0.0, 0.0]
    assert centered[1] == [-3.0, -4.0]


@pytest.mark.asyncio
async def test_learn_from_rag_empty(agents_db):
    normalizer = await Normalizer.learn_from_rag(agents_db, model="non-existent/model")
    assert normalizer.center_vector is None

    embeddings = [[1.0, 2.0]]
    assert normalizer.transform(embeddings) == embeddings


@pytest.mark.asyncio
async def test_rag_service_learn_normalizer(agents_db):
    model = "openai/test-model"
    # We need to mock compute_embeddings or ensure it's not called if we don't want to hit real API.
    # But for this test, we can just manually insert into RagIndex.

    embeddings = [[1.0, 1.0], [3.0, 3.0]]
    for i, emb in enumerate(embeddings):
        idx = RagIndex(
            model=model,
            collection_name="test",
            source_id=f"s{i}",
            embedding=emb,
            embedding_size=2,
        )
        agents_db.add(idx)
    await agents_db.commit()

    rag_service = RagService(uri_or_session=agents_db, model=model)
    normalizer = await rag_service.learn_normalizer(collection_name="test")

    assert np.allclose(normalizer.center_vector, [2.0, 2.0])


def test_get_default_normalizer(tmp_path, monkeypatch):
    from kavalai.normalizer import get_default_normalizer
    import kavalai.normalizer

    # Clear cache for testing
    kavalai.normalizer._default_normalizer = None

    # 1. Default (no env var)
    normalizer = get_default_normalizer()
    assert normalizer.l2 is True
    assert normalizer.center_enabled is False

    # 2. With env var
    kavalai.normalizer._default_normalizer = None
    yaml_path = os.path.join(tmp_path, "custom_normalizer.yaml")
    custom = Normalizer(l1=True, l2=False, center=True, center_vector=[0.1, 0.2])
    custom.save_to_yaml(yaml_path)

    monkeypatch.setenv("KAVALAI_EMBEDDING_NORMALIZER_YAML", yaml_path)
    normalizer2 = get_default_normalizer()
    assert normalizer2.l1 is True
    assert normalizer2.l2 is False
    assert normalizer2.center_enabled is True
    assert np.allclose(normalizer2.center_vector, [0.1, 0.2])

    # 3. Caching
    normalizer3 = get_default_normalizer()
    assert normalizer3 is normalizer2

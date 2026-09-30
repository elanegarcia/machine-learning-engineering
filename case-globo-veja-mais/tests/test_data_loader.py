import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset


def _write_csv(tmp_path, rows):
    """Helper: escreve um CSV de teste no formato esperado (url, title, embedding)."""
    df = pd.DataFrame(rows)
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_dataset_happy_path(tmp_path):
    rows = [
        {"url": "http://a.com/1", "title": "Matéria 1", "embedding": str([0.1, 0.2, 0.3])},
        {"url": "http://a.com/2", "title": "Matéria 2", "embedding": str([0.4, 0.5, 0.6])},
    ]
    path = _write_csv(tmp_path, rows)

    df, embeddings = load_dataset(path)

    assert list(df.columns) == ["url", "title"]
    assert len(df) == 2
    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings[0], [0.1, 0.2, 0.3], atol=1e-6)


def test_load_dataset_missing_column_raises(tmp_path):
    rows = [{"url": "http://a.com/1", "embedding": str([0.1, 0.2, 0.3])}]  # falta "title"
    path = _write_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_dataset(path)


def test_load_dataset_inconsistent_embedding_dims_raises(tmp_path):
    rows = [
        {"url": "http://a.com/1", "title": "Matéria 1", "embedding": str([0.1, 0.2, 0.3])},
        {"url": "http://a.com/2", "title": "Matéria 2", "embedding": str([0.1, 0.2])},  # dimensão diferente
    ]
    path = _write_csv(tmp_path, rows)

    with pytest.raises(ValueError, match="dimensões inconsistentes"):
        load_dataset(path)


def test_load_dataset_preserves_row_order(tmp_path):
    rows = [
        {"url": f"http://a.com/{i}", "title": f"Matéria {i}", "embedding": str([float(i), 0.0])}
        for i in range(5)
    ]
    path = _write_csv(tmp_path, rows)

    df, embeddings = load_dataset(path)

    assert list(df["url"]) == [f"http://a.com/{i}" for i in range(5)]
    np.testing.assert_allclose(embeddings[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
import numpy as np

from src.preprocessing import center_and_normalize


def test_center_and_normalize_mean_is_near_zero():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(loc=5.0, scale=2.0, size=(50, 16))

    result = center_and_normalize(embeddings)

    # a média dos vetores CENTRALIZADOS (antes de normalizar) deve ser ~0;
    # aqui valido indiretamente checando que o resultado não herda o offset original
    assert abs(result.mean()) < 0.5


def test_center_and_normalize_unit_norm():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(30, 8))

    result = center_and_normalize(embeddings)
    norms = np.linalg.norm(result, axis=1)

    np.testing.assert_allclose(norms, np.ones(30), atol=1e-5)


def test_center_and_normalize_output_shape_matches_input():
    embeddings = np.random.default_rng(2).normal(size=(10, 768))

    result = center_and_normalize(embeddings)

    assert result.shape == embeddings.shape


def test_center_and_normalize_handles_zero_vector_after_centering():
    # um vetor idêntico à média do dataset fica com norma 0 após centralizar;
    # a função não pode levantar erro de divisão por zero nesse caso
    embeddings = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [3.0, -1.0],
    ])

    result = center_and_normalize(embeddings)

    assert np.all(np.isfinite(result))


def test_center_and_normalize_reduces_average_pairwise_similarity():
    # replica o achado da EDA: em um espaço anisotrópico, centralizar reduz
    # a similaridade média entre pares aleatórios de itens
    rng = np.random.default_rng(3)
    base = rng.normal(size=(1, 32))
    noise = rng.normal(scale=0.3, size=(100, 32))
    embeddings = base + noise  # todos os vetores próximos de uma direção comum

    raw_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    raw_sim = raw_normalized @ raw_normalized.T
    np.fill_diagonal(raw_sim, np.nan)

    centered = center_and_normalize(embeddings)
    centered_sim = centered @ centered.T
    np.fill_diagonal(centered_sim, np.nan)

    assert np.nanmean(centered_sim) < np.nanmean(raw_sim)
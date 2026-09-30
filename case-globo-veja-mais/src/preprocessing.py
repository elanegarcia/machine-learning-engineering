import numpy as np

def center_and_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Centraliza pela média do dataset e normaliza L2 (cosseno = produto interno)."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # evita divisão por zero em caso extremo
    return centered / norms
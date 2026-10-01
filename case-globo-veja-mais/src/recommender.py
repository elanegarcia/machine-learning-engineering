# Recomenda o núcleo de recomendação

import json
import numpy as np
import pandas as pd

def recommend_for_index(
    idx: int, X: np.ndarray, urls: pd.Series, k: int = 10
) -> list[list]:
    """Retorna os k vizinhos mais similares ao item em `idx`, excluindo ele mesmo."""
    sims = X @ X[idx]
    sims[idx] = -np.inf
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [[urls.iloc[j], round(float(sims[j]), 4)] for j in top_idx]


def build_recommendations(
    df: pd.DataFrame, X: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """Gera o DataFrame final no formato exigido: url, recommended_urls (JSON)."""
    records = []
    for i, url in enumerate(df["url"]):
        recs = recommend_for_index(i, X, df["url"], k=k)
        records.append({"url": url, "recommended_urls": json.dumps(recs)})
    return pd.DataFrame(records)
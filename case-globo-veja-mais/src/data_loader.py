import ast
import numpy as np
import pandas as pd

def load_dataset(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega o CSV e retorna o DataFrame (url, title) e a matriz de embeddings."""
    df = pd.read_csv(path)
    required_cols = {"url", "title", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    parsed = [ast.literal_eval(e) for e in df["embedding"]]
    if len({len(v) for v in parsed}) > 1:
        raise ValueError("Embeddings com dimensões inconsistentes.")

    embeddings = np.array(parsed, dtype=np.float32)
    return df[["url", "title"]].reset_index(drop=True), embeddings
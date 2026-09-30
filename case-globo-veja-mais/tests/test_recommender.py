# tests/test_recommender.py

import numpy as np
import pandas as pd
from src.recommender import build_recommendations
import json

def test_recommendations_shape_and_rules():
    df = pd.DataFrame({"url": [f"u{i}" for i in range(20)]})
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 8))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    result = build_recommendations(df, X, k=10)

    assert len(result) == 20
    for _, row in result.iterrows():
        recs = json.loads(row["recommended_urls"])
        assert len(recs) == 10                            # sempre 10 itens
        rec_urls = [r[0] for r in recs]
        assert row["url"] not in rec_urls                 # nunca recomenda a si mesma
        assert len(rec_urls) == len(set(rec_urls))        # sem duplicatas
        scores = [r[1] for r in recs]
        assert scores == sorted(scores, reverse=True)     # ordenado por score
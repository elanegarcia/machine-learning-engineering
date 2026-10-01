# o src/recommender.py faz a depuração oficial, no entanto, para caso de necessidade de confirmação, realizei este script para verificar se o resultado do recommender.py está correto
# esse script serve como uma verificação adicional.

import sys
sys.path.insert(0, "src")

from data_loader import load_dataset
from preprocessing import center_and_normalize

df, embeddings = load_dataset("data/dataset_rec.csv")
X = center_and_normalize(embeddings)

url_referencia = "http://gshow.globo.com/tudo-mais/tv-e-famosos/noticia/maira-cardi-mostra-novos-prints-sobre-treta-com-mulher-de-thiago-silva.ghtml"

idx = df.index[df["url"] == url_referencia][0]
print("MATÉRIA DE REFERÊNCIA:")
print(df["title"].iloc[idx])
print()

sims = X @ X[idx]
sims[idx] = -float("inf")

top5 = sims.argsort()[::-1][:5]

print("TOP 5 RECOMENDAÇÕES:")
for j in top5:
    print(f"{sims[j]:.4f}  {df['title'].iloc[j]}")
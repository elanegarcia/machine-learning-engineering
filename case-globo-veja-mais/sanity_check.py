import pandas as pd
import json

df = pd.read_csv('output/recommendations.csv')

problemas = []
for _, row in df.iterrows():
    recs = json.loads(row['recommended_urls'])
    urls = [r[0] for r in recs]
    if len(recs) != 10:
        problemas.append(f"{row['url']}: não tem 10 itens")
    if row['url'] in urls:
        problemas.append(f"{row['url']}: recomenda a si mesma")
    if len(urls) != len(set(urls)):
        problemas.append(f"{row['url']}: tem duplicata")
    scores = [r[1] for r in recs]
    if scores != sorted(scores, reverse=True):
        problemas.append(f"{row['url']}: não está ordenado")

print('Problemas encontrados:', len(problemas))
for p in problemas[:10]:
    print(' -', p)
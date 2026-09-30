# Desafio Técnico para a vaga de Engenharia de Machine Learning Pleno - Data & AI

---

## Leia Mais g1 — Recomendação "Veja Mais" (Content-Based)

## Contexto

O time de recomendação do g1 mantém a oferta **Leia Mais**, que sugere matérias para a pessoa usuária com base no que ela está lendo no momento. Este projeto endereça especificamente o componente **"Veja Mais"**, que toma como base a matéria atual e recomenda matérias similares a ela.

Exemplo de matéria de referência:
`https://g1.globo.com/pe/caruaru-regiao/noticia/2024/09/11/deolane-bezerra-segue-presa-apos-audiencia-de-custodia.ghtml`

Como o g1 já possui outras ofertas de recomendação (por exemplo, baseadas em perfil e consumo da pessoa usuária), o "Veja Mais" foi definido como uma oferta **exclusivamente content-based**, para não sobrepor as demais: a recomendação depende apenas do conteúdo da matéria, não de quem está lendo.

## Tarefa

Com base no dataset fornecido, criar uma função em Python que, dada a matéria que a pessoa usuária está lendo, retorne uma lista de matérias similares para exibição no "Veja Mais".

**Requisitos do enunciado:**

- Cada matéria é identificada pela sua `url`.
- A lista de recomendações deve conter **10 itens** por matéria.
- A oferta deve considerar **apenas similaridade de conteúdo** (aspectos de perfil e consumo da pessoa usuária ficam fora de escopo).

**Premissas fornecidas:**

1. O dataset de entrada traz as informações textuais das matérias, com a estrutura:
   - `url: str`
   - `title: str`
   - `embedding: List[float]`
2. A saída da solução deve ser um arquivo CSV com a estrutura:
   - `url: str`
   - `recommended_urls: List[List[str, float]]`

## Dados

- `dataset_rec.csv`: 100 matérias, colunas `url`, `title`, `embedding` (vetor de 768 dimensões, salvo como string).
- Sem valores nulos e sem URLs duplicadas.

### Achados da análise exploratória

- Os embeddings **não vêm normalizados** (norma L2 varia de ~6,0 a ~7,8).
- O espaço é **anisotrópico**: o cosseno médio entre pares aleatórios de matérias é ~0,66 (mínimo ~0,46), ou seja, tudo tende a parecer parecido com tudo, o que reduz o poder discriminativo do ranking bruto.
- **Centralizando** os vetores (subtraindo o vetor médio) antes de normalizar, o cosseno médio cai para próximo de 0, e a ordenação passa a refletir melhor a similaridade temática real (validado por inspeção qualitativa dos vizinhos mais próximos).
- Por isso, a solução usa **centralização + normalização L2** antes do cálculo de similaridade por produto interno (cosseno).

## Abordagem

1. Carregar `dataset_rec.csv` e parsear os embeddings de string para vetores numéricos.
2. Centralizar (subtrair a média do dataset) e normalizar L2 os vetores.
3. Calcular a similaridade por cosseno entre a matéria de referência e todas as demais.
4. Excluir a própria matéria do ranking.
5. Retornar as top 10 recomendações como pares `[url, score]`, ordenadas por score decrescente.

Com apenas 100 itens, uma matriz de similaridade completa (`X @ X.T`) é suficiente e simples. Em escala de produção (milhões de matérias), o mesmo pipeline de pré-processamento seria mantido, mas a busca por vizinhos mais próximos passaria para uma solução aproximada (ANN), como FAISS, ScaNN ou HNSW, com os vizinhos pré-computados em batch e servidos via cache/lookup.

## Formato de saída

`recommendations.csv` com as colunas:

| coluna | tipo | descrição |
|---|---|---|
| `url` | str | URL da matéria de referência |
| `recommended_urls` | str (JSON) | lista de 10 pares `[url, score]`, ordenada por score decrescente |

`recommended_urls` é serializada como JSON (não como lista literal do Python), para evitar ambiguidade de parsing por quem for consumir o CSV. Para ler:

```python
import json
recs = json.loads(row["recommended_urls"])  # -> [[url, score], ...]
```

## Estrutura do repositório

```
.
├── README.md
├── recommender.py       # funções de carga, pré-processamento e recomendação
├── run.py                # gera recommendations.csv a partir do dataset_rec.csv
├── tests/
│   └── test_recommender.py
├── analysis.ipynb        # EDA e comparação de variantes (cru vs. centralizado)
├── dataset_rec.csv
└── recommendations.csv   # saída gerada
```

## Como rodar

```bash
pip install -r requirements.txt   # pandas, numpy
python run.py --input dataset_rec.csv --output recommendations.csv --k 10
```

## Como validar sem ground truth

Não há rótulos de relevância no dataset, então a validação combina:

- **Inspeção qualitativa**: amostragem de matérias de referência e seus top-10, checando se o tema bate.
- **Comparação de variantes**: cosseno cru vs. cosseno com vetores centralizados, para justificar a escolha do pré-processamento.
- **Métricas proxy**:
  - *coverage*: proporção de matérias do catálogo que aparecem em ao menos uma lista de recomendação;
  - *hubness*: distribuição de quantas vezes cada matéria é recomendada, para identificar itens que dominam os rankings de forma desproporcional.
- **Testes automatizados**: cada lista tem exatamente 10 itens, não contém a própria matéria, não tem duplicatas e está ordenada por score decrescente.

## Limitações e próximos passos

- **Near-duplicates**: alguns pares de matérias têm cosseno muito alto (~0,97), indicando conteúdo quase idêntico (ex.: coberturas do mesmo fato). Dependendo do objetivo de produto, isso pode ser desejável ou não; um re-ranking com **MMR (Maximal Marginal Relevance)** permitiria balancear relevância e diversidade via um parâmetro ajustável.
- **Cold start**: matérias novas, sem embedding pré-calculado, precisam ser embedadas no momento da publicação antes de entrarem no índice de recomendação.
- **Recência**: o dataset não traz data de publicação; em produção, é esperado que a recência da matéria influencie o ranking, já que conteúdo jornalístico perde relevância com o tempo.
- **Atualização do índice**: definir a frequência de recálculo das recomendações conforme o volume de publicações do g1.
- **Avaliação online**: complementar a validação offline com testes A/B (CTR no componente "Veja Mais") após o deploy.
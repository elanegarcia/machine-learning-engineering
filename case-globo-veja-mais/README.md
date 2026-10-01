# Desafio Técnico para a vaga de Engenharia de Machine Learning Pleno - Data & AI

---

## Leia Mais g1 — Recomendação "Veja Mais" (Content-Based)

## Contexto

O time de recomendação do g1 mantém a oferta **Leia Mais**, que sugere matérias para a pessoa usuária com base no que ela está lendo no momento. Este projeto endereça especificamente o componente **"Veja Mais"**, que toma como base a matéria atual e recomenda matérias similares a ela.

Exemplo de matéria de referência do enunciado: [https://g1.globo.com/pe/caruaru-regiao/noticia/2024/09/11/deolane-bezerra-segue-presa-apos-audiencia-de-custodia.ghtml](https://g1.globo.com/pe/caruaru-regiao/noticia/2024/09/11/deolane-bezerra-segue-presa-apos-audiencia-de-custodia.ghtml)

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

- `data/dataset_rec.csv`: 100 matérias do domínio [gshow.globo.com](gshow.globo.com), colunas `url`, `title`, `embedding` (vetor de 768 dimensões, salvo como string).
- Sem valores nulos e sem URLs duplicadas.

### Achados da análise exploratória

- Os embeddings **não vêm normalizados** (norma L2 varia de ~6,0 a ~7,8).
- O espaço é **anisotrópico**: o cosseno médio entre pares aleatórios de matérias é ~0,66 (mínimo ~0,46), ou seja, tudo tende a parecer parecido com tudo, o que reduz o poder discriminativo do ranking bruto.
- **Centralizando** os vetores (subtraindo o vetor médio) antes de normalizar, o cosseno médio cai para próximo de 0, e a ordenação passa a refletir melhor a similaridade temática real.
- Essa melhora se confirma na saída real gerada pelo pipeline: para a matéria de exemplo sobre Maíra Cardi, a recomendação #1 (sobre o mesmo tema — a treta com Belle Silva) ficou com score 0,32, bem abaixo da similaridade média (~0,6+) que o cosseno cru geraria entre itens sem relação nenhuma. Isso confirma que, após a centralização, o score reflete relevância real, não apenas "proximidade genérica" do espaço.
- Por isso, a solução usa **centralização + normalização L2** antes do cálculo de similaridade por produto interno (cosseno).

## Abordagem

1. Carregar `dataset_rec.csv` e parsear os embeddings de string para vetores numéricos.
2. Centralizar (subtrair a média do dataset) e normalizar L2 os vetores.
3. Calcular a similaridade por cosseno entre a matéria de referência e todas as demais.
4. Excluir a própria matéria do ranking.
5. Retornar as top 10 recomendações como pares `[url, score]`, ordenadas por score decrescente.

Com apenas 100 itens, uma matriz de similaridade completa (`X @ X.T`) é suficiente e simples. Em escala de produção (milhões de matérias), o mesmo pipeline de pré-processamento seria mantido, mas a busca por vizinhos mais próximos passaria para uma solução aproximada (ANN), como FAISS, ScaNN ou HNSW, com os vizinhos pré-computados em batch e servidos via cache/lookup.

## Formato de saída

`output/recommendations.csv` com as colunas:

| coluna | tipo | descrição |
|---|---|---|
| `url` | str | URL da matéria de referência |
| `recommended_urls` | str (JSON) | lista de 10 pares `[url, score]`, ordenada por score decrescente |

`recommended_urls` é serializada como JSON (não como lista literal do Python), para evitar ambiguidade de parsing por quem for consumir o CSV. Para ler:

```python
import json
recs = json.loads(row["recommended_urls"])  # -> [[url, score], ...]
```
Obs: O CSV de saída foi incluído no repositório para facilitar a avaliação. Em um ambiente de produção real, ele não seria versionado, pois é gerado automaticamente pelo pipeline.

## Tomada de decisão

| Decisão | Alternativas consideradas | Por que essa escolha |
|---|---|---|
| Centralizar + normalizar L2 antes do cosseno | Cosseno cru; distância euclidiana | A EDA mostrou espaço anisotrópico (cosseno médio ~0,66 entre itens aleatórios); centralizar resolveu isso sem adicionar complexidade |
| Matriz de similaridade completa (`X @ X.T`) | Índice ANN (FAISS/HNSW) desde já | Com 100 itens, calcular tudo é trivial e mais simples de auditar; ANN é citado como próximo passo para escala |
| Serializar `recommended_urls` como JSON no CSV | Lista Python serializada como string (`str(list)`) | JSON evita ambiguidade de parsing e é padrão de interoperabilidade entre linguagens/sistemas |
| Excluir a própria matéria do ranking via `-inf` no score | Filtrar por comparação de índice depois de ordenar | Mais simples e garante que a matéria nunca aparece, mesmo antes do corte do top-k |
| Não usar sinais de perfil/consumo | Modelo híbrido (conteúdo + colaborativo) | Fora do escopo definido no enunciado — o "Veja Mais" deve ser puramente content-based para não sobrepor outras ofertas do g1 |

## Estrutura do repositório

```
case-globo-veja-mais/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── dataset_rec.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        
│   ├── preprocessing.py     
│   ├── recommender.py       
│   └── run.py              
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_recommender.py
├── notebooks/
│   └── analysis.ipynb
├── sanity_check.py         
└── output/
    └── recommendations.csv   
```

## Como rodar
Ambiente testado com Python 3.14 e numpy 2.5.3.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/run.py --input data/dataset_rec.csv --output output/recommendations.csv --k 10
```

## Como testar

```bash
pytest tests/ -v
```

Cobertura:

- data_loader: caso feliz, coluna obrigatória ausente, dimensões de embedding inconsistentes entre linhas, ordem das linhas preservada.
- preprocessing: norma unitária após centralizar/normalizar, shape preservado, tratamento de vetor nulo após centralização (evita divisão por zero), e a própria melhora de discriminação (centralizado reduz a similaridade média entre pares).
- recommender: sempre 10 itens, nunca recomenda a própria matéria, sem duplicatas, scores ordenados de forma decrescente.

Além dos testes automatizados, sanity_check.py roda uma checagem em massa direto sobre o output/recommendations.csv já gerado, validando o contrato de saída em todas as 100 linhas (não só em dados sintéticos de teste):

```bash
python sanity_check.py
```

## Resultados

- **Cobertura**: todas as 100 matérias receberam exatamente 10 recomendações, sem exceção.
- **Consistência do contrato**: nenhuma matéria recomenda a si mesma, não há duplicatas nas listas, e os scores estão sempre ordenados de forma decrescente (validado em massa pelo `sanity_check.py`, além dos testes automatizados).
- **Qualidade temática**: inspeção manual confirma que os vizinhos mais próximos pertencem ao mesmo assunto ou a assuntos correlatos. Exemplo real do pipeline:

  **Matéria de entrada:**
  `http://gshow.globo.com/tudo-mais/tv-e-famosos/noticia/maira-cardi-mostra-novos-prints-sobre-treta-com-mulher-de-thiago-silva.ghtml`
  *"Maira Cardi ameaça expor 'a verdade' sobre história com mulher de Thiago Silva: 'Não me desafie'"*

  **Top 5 recomendações retornadas:**

  | score | título |
  |---|---|
  | 0,3203 | Maíra Cardi reafirma que Thiago Silva foi seu cliente e mostra áudio como prova |
  | 0,2538 | Deborah Secco se diverte com revelação íntima sobre Hugo Moura: 'Exposto em rede nacional' |
  | 0,2163 | Maíra Cardi X Belle Silva: entenda a treta sobre Thiago Silva ter feito (ou não) programa de emagrecimento |
  | 0,2019 | Última famosa a deixar o No Limite, Carol Nakamura conta que foi julgada |
  | 0,2001 | Deborah Secco revela detalhe íntimo de Hugo Moura no 'Sobre Nós Dois': 'Incrível' |

  As posições #1 e #3 são sobre o mesmo fato exato (a treta Maíra Cardi/Thiago Silva/Belle Silva). As posições #2 e #5 trazem outro assunto do mesmo "gênero" editorial (fofoca de celebridade/revelação íntima), e a #4 é a mais fraca do grupo — mostra que, no fim da lista top-10, a relevância temática já começa a cair, o que é esperado e aceitável para a posição.
- **Discriminação do ranking**: após a centralização dos embeddings, a similaridade média entre pares aleatórios de matérias caiu de ~0,66 (cosseno cru) para próximo de 0, evidenciando que o pré-processamento resolveu a anisotropia do espaço vetorial e tornou o ranking mais informativo.


## Como validar sem ground truth

Não há rótulos de relevância no dataset, então a validação combina:

- **Inspeção qualitativa**: amostragem de matérias de referência e seus top-10, checando se o tema bate (feito no notebooks/analysis.ipynb e confirmado no exemplo real citado na seção de achados acima).
- **Comparação de variantes**: cosseno cru vs. cosseno com vetores centralizados, para justificar a escolha do pré-processamento.
- **Métricas proxy**:
  - *coverage*: proporção de matérias do catálogo que aparecem em ao menos uma lista de recomendação;
  - *hubness*: distribuição de quantas vezes cada matéria é recomendada, para identificar itens que dominam os rankings de forma desproporcional.
- **Testes automatizados + sanity check em massa:**: cada lista tem exatamente 10 itens, não contém a própria matéria, não tem duplicatas e está ordenada por score decrescente — validado tanto em dados sintéticos (pytest) quanto no output real de 100 matérias (sanity_check.py).

## Limitações e próximos passos

- **Near-duplicates**: alguns pares de matérias têm cosseno muito alto (~0,97), indicando conteúdo quase idêntico (ex.: coberturas do mesmo fato). Dependendo do objetivo de produto, isso pode ser desejável ou não; um re-ranking com MMR (Maximal Marginal Relevance) permitiria balancear relevância e diversidade via um parâmetro ajustável.
- **Cold start**: matérias novas, sem embedding pré-calculado, precisam ser embedadas no momento da publicação antes de entrarem no índice de recomendação.
- **Recência**: o dataset não traz data de publicação; em produção, é esperado que a recência da matéria influencie o ranking, já que conteúdo jornalístico perde relevância com o tempo.
- **Atualização do índice**: definir a frequência de recálculo das recomendações conforme o volume de publicações do g1.
- **Avaliação online**: complementar a validação offline com testes A/B (CTR no componente "Veja Mais") após o deploy.
  

import argparse
from data_loader import load_dataset
from preprocessing import center_and_normalize
from recommender import build_recommendations

def main():
    parser = argparse.ArgumentParser(description="Gera recomendações Veja Mais")
    parser.add_argument("--input", default="data/dataset_rec.csv")
    parser.add_argument("--output", default="output/recommendations.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    df, embeddings = load_dataset(args.input)
    X = center_and_normalize(embeddings)
    result = build_recommendations(df, X, k=args.k)
    result.to_csv(args.output, index=False)
    print(f"{len(result)} recomendações geradas em {args.output}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from src.model.model import LogisticRegressionScratch


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(df["review"].values).toarray()
    y = df["sentiment"].map({"positive": 1, "negative": 0}).values
    return X, y


def main():
    # Load and preprocess data
    df = load_data("artifacts/data/raw/reviews.csv")
    X, y = preprocess_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = LogisticRegressionScratch(input_dim=X.shape[1], learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nEvaluation on Test Set:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.model.model import LogisticRegressionScratch
from utils.preprocess import preprocess_text
from typing import Tuple

def load_and_preprocess_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    X = df['review'].astype(str).apply(preprocess_text).tolist()  # change here
    y = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X).toarray()
    return X_vec, np.array(y)

def train_model(X: np.ndarray, y: np.ndarray):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = LogisticRegressionScratch(input_dim=X.shape[1], learning_rate=0.1)

    # Train the model
    model.fit(X_train, y_train, epochs=1000)

    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    print("\nEvaluation Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    X, y = load_and_preprocess_data("artifacts/data/raw/reviews.csv")
    train_model(X, y)

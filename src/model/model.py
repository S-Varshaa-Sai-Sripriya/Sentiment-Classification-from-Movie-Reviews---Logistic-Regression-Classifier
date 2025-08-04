import numpy as np
from typing import List, Dict


class LogisticRegressionScratch:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.01, 0.01, size=input_dim)
        self.bias = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True) -> None:
        n_samples, _ = X.shape

        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)

            loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))

            dw = np.dot(X.T, (y_hat - y)) / n_samples
            db = np.sum(y_hat - y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_probabilities(X)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_true = y

        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = (TP + TN) / len(y_true)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

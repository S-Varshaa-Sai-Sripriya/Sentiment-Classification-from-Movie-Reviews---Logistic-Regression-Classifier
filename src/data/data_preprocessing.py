import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import logger
from utils.exception import CustomException

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing punctuation, lowercasing, and extra spaces.

    Args:
        text (str): Raw review text.

    Returns:
        str: Cleaned review text.
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_and_save(file_path: str, output_dir: str) -> None:
    """
    Preprocesses the dataset and saves processed TF-IDF features and labels.

    Args:
        file_path (str): Path to the raw CSV file.
        output_dir (str): Path where processed data will be stored.
    """
    try:
        df = pd.read_csv(file_path)
        df["cleaned_review"] = df["review"].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(df["cleaned_review"])
        y = df["sentiment"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        with open(output_dir_path / "X_train.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open(output_dir_path / "X_test.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with open(output_dir_path / "y_train.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open(output_dir_path / "y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)
        with open(output_dir_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info("✅ Data preprocessing completed and files saved.")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise CustomException(e)

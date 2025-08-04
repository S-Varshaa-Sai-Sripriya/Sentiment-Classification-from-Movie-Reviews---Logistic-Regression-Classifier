from pathlib import Path
import pandas as pd
from utils.logger import logger
from utils.exception import CustomException

def validate_data(file_path: str) -> bool:
    """
    Validates the structure and content of the dataset.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        bool: True if validation is successful, else raises CustomException.
    """
    try:
        file = Path(file_path)
        if not file.exists():
            raise CustomException("Data file does not exist.")

        df = pd.read_csv(file)

        # Basic structure check
        expected_columns = {"review", "sentiment"}
        if not expected_columns.issubset(df.columns):
            raise CustomException(f"Missing required columns. Found: {df.columns.tolist()}")

        # Null value check
        if df.isnull().sum().any():
            raise CustomException("Dataset contains null values.")

        # Value type check for 'sentiment'
        if not df["sentiment"].isin([0, 1]).all():
            raise CustomException("Sentiment column must contain only 0 or 1.")

        logger.info("✅ Data validation passed.")
        return True

    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        raise CustomException(e)

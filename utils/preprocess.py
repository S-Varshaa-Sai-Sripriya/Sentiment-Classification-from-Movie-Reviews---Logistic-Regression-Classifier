import re
import string

def preprocess_text(text: str) -> str:
    """
    Cleans the input text by:
    - Lowercasing
    - Removing punctuation
    - Removing extra whitespace
    - Removing numbers (optional)

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove digits (optional)
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text

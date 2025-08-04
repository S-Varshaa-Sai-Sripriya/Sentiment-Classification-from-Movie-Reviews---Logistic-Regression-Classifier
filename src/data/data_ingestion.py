import os
import random
import pandas as pd
from faker import Faker

fake = Faker()

POSITIVE_TEMPLATES = [
    "I absolutely loved this movie. {}",
    "What a fantastic experience! {}",
    "Brilliant storyline and strong performances. {}",
    "This is one of the best movies I've seen. {}",
    "Totally worth the watch. {}"
]

NEGATIVE_TEMPLATES = [
    "I really hated this movie. {}",
    "Terrible plot and poor acting. {}",
    "What a waste of time. {}",
    "I do not recommend watching this. {}",
    "Disappointing and boring. {}"
]

ADDITIONAL_POSITIVE_PHRASES = [
    "Highly recommended!", "Oscar-worthy performance.", "Great direction and cinematography.",
    "Loved the characters.", "A must-watch film."
]

ADDITIONAL_NEGATIVE_PHRASES = [
    "Fell asleep halfway.", "Terribly edited.", "Very predictable plot.",
    "Bad acting ruined it.", "Unrealistic and annoying characters."
]

def generate_review(sentiment: str) -> str:
    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
        addon = random.choice(ADDITIONAL_POSITIVE_PHRASES)
    else:
        template = random.choice(NEGATIVE_TEMPLATES)
        addon = random.choice(ADDITIONAL_NEGATIVE_PHRASES)
    return template.format(addon)

def generate_dataset(num_samples: int = 1000, output_path: str = "artifacts/data/raw/reviews.csv") -> None:
    reviews = []
    sentiments = []

    for _ in range(num_samples):
        sentiment = random.choice(["positive", "negative"])
        review = generate_review(sentiment)
        reviews.append(review)
        sentiments.append(sentiment)

    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset with {num_samples} samples saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()

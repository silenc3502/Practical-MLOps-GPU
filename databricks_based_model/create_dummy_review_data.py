import pandas as pd
import random

def get_dummy_data(n=1000):
    reviews = ["좋아요", "별로예요", "보통이에요", "정말 좋아요", "싫어요", "괜찮아요"]
    sentiments = ["positive", "negative", "neutral"]

    data = {
        "id": list(range(n)),
        "review": [random.choice(reviews) for _ in range(n)],
        "sentiment": [random.choice(sentiments) for _ in range(n)]
    }
    return pd.DataFrame(data)

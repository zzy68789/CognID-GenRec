from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfItemTextEncoder:
    def __init__(self, max_features: int = 2048) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit_transform(self, items: pd.DataFrame):
        texts = items.apply(_join_item_text, axis=1).tolist()
        return self.vectorizer.fit_transform(texts)


def _join_item_text(row: pd.Series) -> str:
    fields = [
        str(row.get("title", "")),
        str(row.get("body", "")),
        str(row.get("topic", "")),
    ]
    return " ".join(field for field in fields if field and field != "nan")

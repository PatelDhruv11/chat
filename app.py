"""Simple AI app in Python: a tiny command-line sentiment classifier.

This script trains a Naive Bayes classifier on a small built-in dataset and
predicts whether user text is positive or negative.
"""

from __future__ import annotations

import math
import re
from collections import Counter


TRAINING_DATA = [
    ("I love this product it is amazing", "positive"),
    ("I love this service and support", "positive"),
    ("This is fantastic and very good", "positive"),
    ("Great experience I am happy", "positive"),
    ("Excellent work and wonderful team", "positive"),
    ("I hate this it is terrible", "negative"),
    ("Bad experience and awful quality", "negative"),
    ("This is disappointing and poor", "negative"),
    ("Horrible service I am unhappy", "negative"),
]


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-z']+", text.lower())


class NaiveBayesSentiment:
    """A tiny multinomial Naive Bayes sentiment model."""

    def __init__(self) -> None:
        self.class_totals = Counter()
        self.word_counts = {"positive": Counter(), "negative": Counter()}
        self.vocab = set()
        self.total_docs = 0

    def fit(self, samples: list[tuple[str, str]]) -> None:
        for text, label in samples:
            self.class_totals[label] += 1
            self.total_docs += 1
            words = tokenize(text)
            self.word_counts[label].update(words)
            self.vocab.update(words)

    def predict_proba(self, text: str) -> dict[str, float]:
        words = tokenize(text)
        scores = {}
        vocab_size = len(self.vocab)

        for label in ["positive", "negative"]:
            # Prior P(label)
            prior = math.log(self.class_totals[label] / self.total_docs)
            # Likelihood with Laplace smoothing
            total_words_in_class = sum(self.word_counts[label].values())
            likelihood = 0.0
            for word in words:
                count = self.word_counts[label][word]
                likelihood += math.log((count + 1) / (total_words_in_class + vocab_size))
            scores[label] = prior + likelihood

        # Convert log scores to normalized probabilities
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    def predict(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)


def main() -> None:
    print("=== Mini AI App: Sentiment Classifier ===")
    print("Type a sentence (or 'quit' to exit).")

    model = NaiveBayesSentiment()
    model.fit(TRAINING_DATA)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter some text.")
            continue

        probs = model.predict_proba(user_input)
        label = model.predict(user_input)
        print(f"AI prediction: {label}")
        print(
            f"Confidence -> positive: {probs['positive']:.2%}, negative: {probs['negative']:.2%}"
        )


if __name__ == "__main__":
    main()

"""Reproducible SMS spam classification pipeline.

The built-in messages are intentionally tiny and are only for demonstrating
the API. Use ``--data`` with a real labelled CSV/TSV file for evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

Message = tuple[str, str]

DEMO_MESSAGES: tuple[Message, ...] = (
    ("Win a free prize now, claim your reward", "spam"),
    ("Congratulations, you have won a cash voucher", "spam"),
    ("Urgent offer, click the link to receive your bonus", "spam"),
    ("Free entry in our weekly competition, reply WIN", "spam"),
    ("You have been selected for a free vacation", "spam"),
    ("Exclusive deal, buy now and save 80 percent", "spam"),
    ("Claim your guaranteed reward before it expires", "spam"),
    ("Limited time promotion, text YES to enter", "spam"),
    ("Are you free for a call at five today?", "ham"),
    ("Please send me the meeting notes when you can", "ham"),
    ("I will reach home around seven", "ham"),
    ("Can we move our appointment to tomorrow?", "ham"),
    ("Thanks for helping with the assignment", "ham"),
    ("Lunch is ready, come downstairs", "ham"),
    ("The train is delayed by ten minutes", "ham"),
    ("Happy birthday! Hope you have a great day", "ham"),
    ("Let me know when you arrive", "ham"),
    ("I have shared the project document with you", "ham"),
    ("Can you call me after class?", "ham"),
)


def load_dataset(path: str | Path) -> list[Message]:
    """Load labelled messages from a CSV or TSV with ``label`` and ``text`` columns."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as data_file:
        sample = data_file.read(2048)
        data_file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        rows = csv.DictReader(data_file, dialect=dialect)
        if not rows.fieldnames or not {"label", "text"}.issubset(rows.fieldnames):
            raise ValueError("Dataset must contain 'label' and 'text' columns.")
        messages = []
        for row_number, row in enumerate(rows, start=2):
            label = (row.get("label") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if label not in {"ham", "spam"} or not text:
                raise ValueError(
                    f"Row {row_number} must have a non-empty text and label 'ham' or 'spam'."
                )
            messages.append((text, label))
    if len(messages) < 4 or len({label for _, label in messages}) < 2:
        raise ValueError("Dataset must contain at least four messages and both labels.")
    return messages


def build_model() -> Pipeline:
    """Create the TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(
    messages: Sequence[Message],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, object]]:
    """Split, train, and evaluate a model with deterministic results."""
    if len(messages) < 4:
        raise ValueError("At least four labelled messages are required.")
    texts, labels = zip(*messages)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=["ham", "spam"], zero_division=0
        ),
    }
    return model, metrics


def predict_message(model: Pipeline, message: str) -> str:
    """Classify one message as spam or ham."""
    if not message.strip():
        raise ValueError("Message must not be empty.")
    return str(model.predict([message])[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Path to a CSV/TSV dataset.")
    source.add_argument(
        "--demo",
        action="store_true",
        help="Run the tiny built-in teaching example (not a meaningful benchmark).",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    messages = list(DEMO_MESSAGES) if args.demo else load_dataset(args.data)
    if args.demo:
        print("DEMONSTRATION ONLY: this tiny dataset is not a meaningful benchmark.")
    model, metrics = train_model(messages, args.test_size, args.random_state)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Spam precision: {metrics['precision']:.3f}")
    print(f"Spam recall: {metrics['recall']:.3f}")
    print(f"Spam F1: {metrics['f1']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; ham, spam):")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])
    for message in (
        "You have won a free vacation, click to claim now",
        "Are you free for a call at 5 pm today?",
    ):
        print(f"{message!r} -> {predict_message(model, message)}")


if __name__ == "__main__":
    main()

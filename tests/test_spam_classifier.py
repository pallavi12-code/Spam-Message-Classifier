import csv

import pytest

from spam_classifier import DEMO_MESSAGES, load_dataset, predict_message, train_model


def test_load_dataset_reads_csv(tmp_path):
    path = tmp_path / "messages.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(
            [
                {"label": "ham", "text": "See you soon"},
                {"label": "spam", "text": "Claim your prize"},
                {"label": "ham", "text": "Call me later"},
                {"label": "spam", "text": "Free offer now"},
            ]
        )

    assert load_dataset(path) == [
        ("See you soon", "ham"),
        ("Claim your prize", "spam"),
        ("Call me later", "ham"),
        ("Free offer now", "spam"),
    ]


def test_load_dataset_rejects_invalid_label(tmp_path):
    path = tmp_path / "messages.tsv"
    path.write_text("label\ttext\nunknown\tHello\nham\tHi\nspam\tOffer\nham\tBye\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Row 2"):
        load_dataset(path)


def test_training_is_reproducible_and_predicts():
    model_one, metrics_one = train_model(DEMO_MESSAGES)
    model_two, metrics_two = train_model(DEMO_MESSAGES)

    assert metrics_one == metrics_two
    assert predict_message(model_one, "Free prize, click now") == "spam"
    assert predict_message(model_two, "Can you call me later?") == "ham"


def test_prediction_rejects_empty_message():
    model, _ = train_model(DEMO_MESSAGES)

    with pytest.raises(ValueError, match="empty"):
        predict_message(model, " ")

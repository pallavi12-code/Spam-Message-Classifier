"""Small, reproducible spam-message classification baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

MESSAGES = [
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
]


def build_model() -> Pipeline:
    """Create the TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])


def train_model(messages=MESSAGES) -> tuple[Pipeline, float]:
    texts, labels = zip(*messages)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = build_model()
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    return model, accuracy


def predict_message(model: Pipeline, message: str) -> str:
    """Classify one message as spam or ham."""
    return str(model.predict([message])[0])


if __name__ == "__main__":
    model, _ = train_model()
    examples = [
        "You have won a free vacation, click to claim now",
        "Are you free for a call at 5 pm today?",
    ]
    print("--- Custom Predictions ---")
    for message in examples:
        print(f"{message!r} -> {predict_message(model, message)}")

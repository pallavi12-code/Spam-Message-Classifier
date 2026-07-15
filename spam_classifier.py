"""
Spam Message Classifier
------------------------
A small ML project that classifies text messages as "spam" or "ham" (not spam)
using a Naive Bayes classifier with TF-IDF features.

Run:
    python spam_classifier.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Sample dataset (small, built-in so no external file needed)
# ---------------------------------------------------------
messages = [
    "Win a free iPhone now, click this link!!!",
    "Congratulations, you have won a $1000 gift card",
    "URGENT: Your account has been suspended, verify now",
    "Claim your free lottery prize today",
    "Limited time offer, buy one get one free",
    "You have been selected for a cash reward, click here",
    "Get rich quick with this one simple trick",
    "Free entry to win a brand new car",
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the notes from today's class?",
    "Don't forget to submit the assignment by Friday",
    "Let's catch up this weekend, it's been a while",
    "The meeting has been rescheduled to 3 PM",
    "Please review the attached report before the call",
    "Happy birthday! Hope you have a great day",
    "Can we reschedule our call to tomorrow morning?",
]

labels = [
    "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

# ---------------------------------------------------------
# 2. Split into train/test sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.25, random_state=42
)

# ---------------------------------------------------------
# 3. Convert text to TF-IDF features
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Train a Naive Bayes classifier
# ---------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. Try your own messages
# ---------------------------------------------------------
def predict_message(text: str) -> str:
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


if __name__ == "__main__":
    sample_inputs = [
        "You have won a free vacation, click to claim now",
        "Are you free for a call at 5 pm today?",
    ]
    print("\n--- Custom Predictions ---")
    for msg in sample_inputs:
        print(f"'{msg}' -> {predict_message(msg)}")

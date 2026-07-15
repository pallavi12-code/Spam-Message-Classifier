# Spam Message Classifier

A small machine learning project that classifies text messages as **spam** or **ham** (not spam) using TF-IDF vectorization and a Naive Bayes classifier (scikit-learn).

## What it does
- Takes a small built-in dataset of spam/ham messages
- Converts text into TF-IDF features
- Trains a Multinomial Naive Bayes model
- Evaluates accuracy and prints a classification report
- Lets you test custom messages via `predict_message()`

## Tech Stack
- Python
- scikit-learn

## How to run

```bash
pip install -r requirements.txt
python spam_classifier.py
```

## Sample Output
```
Accuracy: 0.50

Classification Report:
              precision    recall  f1-score   support
         ham       ...
        spam       ...

--- Custom Predictions ---
'You have won a free vacation, click to claim now' -> spam
'Are you free for a call at 5 pm today?' -> ham
```

## Future Improvements
- Use a larger real-world dataset (e.g., SMS Spam Collection dataset)
- Try other models (Logistic Regression, SVM)
- Add a simple Streamlit UI for live predictions

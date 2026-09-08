# Spam Message Classifier

A small, reproducible NLP baseline that classifies SMS-style text as **spam** or **ham** using TF-IDF features and Multinomial Naive Bayes.

## Pipeline

```text
Message text
   ↓
TF-IDF (unigrams + bigrams)
   ↓
Multinomial Naive Bayes
   ↓
spam / ham
```

## What it demonstrates

- Text preprocessing through `TfidfVectorizer`
- Train/test splitting with stratification
- A scikit-learn `Pipeline` for reproducible preprocessing + modeling
- Accuracy and classification-report evaluation
- Reusable single-message prediction

The repository intentionally uses a small built-in teaching dataset. The reported score should **not** be interpreted as production performance. A real SMS dataset and broader evaluation are the next steps.

## Run locally

```bash
git clone https://github.com/pallavi12-code/Spam-Message-Classifier.git
cd Spam-Message-Classifier
pip install -r requirements.txt
python spam_classifier.py
```

## Example

```text
'You have won a free vacation, click to claim now' -> spam
'Are you free for a call at 5 pm today?' -> ham
```

## Tech stack

- Python
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## Future improvements

- Train and evaluate on the SMS Spam Collection dataset
- Compare Naive Bayes with Logistic Regression and linear SVM
- Add precision/recall and confusion-matrix reporting
- Add a lightweight prediction API or Streamlit interface

## Author

**Pallavi Reddy** — AI & Machine Learning Engineering Student, CBIT

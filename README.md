# Spam Message Classifier

An educational, reproducible NLP baseline that classifies SMS-style messages
as **spam** or **ham** with TF-IDF features and Multinomial Naive Bayes.

## Dataset format

The training script expects a UTF-8 CSV or TSV file with exactly these
important columns:

```text
label,text
ham,"Can you call me after class?"
spam,"Congratulations, claim your free prize"
```

`label` must be `ham` or `spam`; `text` must be non-empty. The repository
does not include a real dataset. You can download a suitable labelled SMS
dataset such as the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
and convert it to this format without committing the data.

The built-in messages are a **demonstration only**, not a meaningful
benchmark. No performance claim is made for them or for any external dataset.

## Reproducible pipeline

1. Load and validate the labelled CSV/TSV input.
2. Split it into stratified train/test sets with `random_state=42`.
3. Fit a scikit-learn pipeline: lowercase TF-IDF unigrams/bigrams followed by
   Multinomial Naive Bayes.
4. Report accuracy, spam precision, spam recall, spam F1, a confusion matrix,
   and the per-class classification report on the held-out test set.

The split and model are deterministic for the same input, dependency versions,
test size, and random seed. A single hold-out score is not a substitute for
cross-validation or evaluation on a representative, independently collected
test set.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python spam_classifier.py --data path/to/messages.csv
python spam_classifier.py --data path/to/messages.tsv --test-size 0.2 --random-state 42
```

To see the API and output format without a real dataset:

```bash
python spam_classifier.py --demo
```

Run tests with:

```bash
python -m pytest
```

## Project structure

- `spam_classifier.py` - dataset loading, preprocessing, training, evaluation,
  and prediction.
- `tests/` - preprocessing, validation, reproducibility, and prediction tests.
- `.github/workflows/ci.yml` - tests on supported Python versions.

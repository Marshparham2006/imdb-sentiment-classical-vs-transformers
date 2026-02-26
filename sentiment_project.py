import numpy as np
import pandas as pd
import time

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from transformers import pipeline

print("\n========== LOAD DATA ==========")

dataset = load_dataset("imdb")

small_train = dataset["train"].shuffle(seed=42).select(range(5000))
small_test = dataset["test"].shuffle(seed=42).select(range(2000))

train_texts = small_train["text"]
train_labels = small_train["label"]

test_texts = small_test["text"]
test_labels = small_test["label"]

print("Train size:", len(train_texts))
print("Test size:", len(test_texts))

# =====================================================
# TF-IDF + Logistic Regression (Hyperparameter Tuning)
# =====================================================

print("\n========== TFIDF MODEL ==========")

pipeline_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.5, 1, 5]
}

grid = GridSearchCV(
    pipeline_model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

start = time.time()
grid.fit(train_texts, train_labels)
end = time.time()

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("Training Time:", round(end - start, 2), "seconds")

# Overfitting check
train_acc = best_model.score(train_texts, train_labels)
test_acc = best_model.score(test_texts, test_labels)

print("\n--- Overfitting Check ---")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Predictions
tfidf_preds = best_model.predict(test_texts)

print("\nTFIDF Accuracy:", accuracy_score(test_labels, tfidf_preds))

print("\nTFIDF Confusion Matrix:")
print(confusion_matrix(test_labels, tfidf_preds))

# Error Analysis
print("\n--- Sample Errors (TFIDF) ---")
wrong_indices = [i for i in range(len(tfidf_preds)) if tfidf_preds[i] != test_labels[i]]

for i in wrong_indices[:3]:
    print("TEXT:", test_texts[i][:200])
    print("TRUE:", test_labels[i])
    print("PRED:", tfidf_preds[i])
    print("-" * 50)

# =====================================================
# TRANSFORMER (Pretrained)
# =====================================================

print("\n========== TRANSFORMER MODEL ==========")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sample_size = 1000
sample_texts = test_texts[:sample_size]
sample_labels = test_labels[:sample_size]

start = time.time()

predictions = sentiment_model(
    sample_texts,
    truncation=True,
    max_length=256,
    padding=True
)

end = time.time()

pred_labels = [1 if p["label"] == "POSITIVE" else 0 for p in predictions]

transformer_acc = accuracy_score(sample_labels, pred_labels)

print("Transformer Accuracy:", transformer_acc)
print("Inference Time:", round(end - start, 2), "seconds")

print("\nTransformer Confusion Matrix:")
print(confusion_matrix(sample_labels, pred_labels))

# =====================================================
# FINAL COMPARISON
# =====================================================

print("\n========== FINAL COMPARISON ==========")

print("TFIDF Accuracy:", accuracy_score(test_labels, tfidf_preds))
print("Transformer Accuracy:", transformer_acc)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

print("\n========== ROC CURVE ==========")

tfidf_probs = best_model.predict_proba(test_texts)[:, 1]

fpr, tpr, _ = roc_curve(test_labels, tfidf_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"TFIDF (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - TFIDF")
plt.legend()
plt.show()

print("AUC Score:", roc_auc)

from sklearn.model_selection import learning_curve

print("\n========== LEARNING CURVE ==========")

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    train_texts,
    train_labels,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 4)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Train Accuracy")
plt.plot(train_sizes, val_mean, label="Validation Accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - TFIDF")
plt.legend()
plt.show()

print("\n========== SPEED COMPARISON ==========")

print("TFIDF Training Time:", round(end - start, 2), "seconds")
print("Transformer Inference Time (1000 samples):", round(end - start, 2), "seconds")

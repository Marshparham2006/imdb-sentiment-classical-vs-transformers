# imdb-sentiment-classical-vs-transformers
Comparative NLP study on IMDb sentiment analysis using TF-IDF + Logistic Regression vs. Pretrained DistilBERT, including Comparative NLP study on IMDb sentiment analysis using TF-IDF + Logistic Regression vs. Pretrained DistilBERT, including hyperparameter tuning, ROC analysis, learning curves, and performance trade-off evaluation.


🎬 IMDb Sentiment Analysis: Classical ML vs Transformer
 Project Overview  :

This project presents a comparative study between a classical machine learning approach and a modern Transformer-based model for sentiment analysis on the IMDb movie review dataset.

We evaluate:

TF-IDF + Logistic Regression (with hyperparameter tuning)

Pretrained DistilBERT (Transformer-based model)

The goal is to analyze performance, overfitting behavior, classification metrics, and computational trade-offs.

 Dataset :

Dataset: IMDb Movie Reviews

Source: HuggingFace Datasets

5,000 training samples

2,000 test samples

Binary classification:

0 → Negative

1 → Positive

 Models Implemented  :
1️ : TF-IDF + Logistic Regression

Hyperparameter tuning via GridSearchCV

Parameters tuned:

ngram_range

max_features

regularization strength (C)

2️ : DistilBERT (Pretrained Transformer)

Model: distilbert-base-uncased-finetuned-sst-2-english

Used via HuggingFace pipeline

Truncation handling for long sequences

  Evaluation Metrics  :

Accuracy

Confusion Matrix

ROC Curve (AUC)

Learning Curve

Overfitting Analysis

Speed Comparison

 Key Results  :
Model	Accuracy	Notes
TF-IDF + Logistic Regression	0.868	Fast training, slight overfitting
DistilBERT (Pretrained)	0.867	Better contextual understanding, slower inference

AUC Score (TF-IDF): 0.946

 Observations :

Classical ML remains highly competitive on medium-sized datasets.

Transformers offer contextual advantages but come with higher computational cost.

Overfitting was observed in Logistic Regression (Train 0.98 vs Test 0.86).

Accuracy alone is insufficient — AUC and confusion matrix provide deeper insights.

 Conclusion :

This project demonstrates that traditional machine learning approaches can still compete with modern deep learning architectures in certain scenarios, especially when computational efficiency is a priority.

 Tech Stack  :

Python

Scikit-learn

HuggingFace Transformers

Matplotlib

NumPy

Pandas

 Future Improvements :

Fine-tuning Transformer models

Testing Longformer for long reviews

Cross-domain generalization experiments

Deploying as a web app (Streamlit)


if you want to see my blog in Medium :   https://medium.com/@marshparham2/is-classical-machine-learning-still-competitive-e04fa2a905c3

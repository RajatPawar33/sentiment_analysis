Sentiment Analysis with RoBERTa on IMDB Dataset
This project implements a sentiment analysis model using a pre-trained RoBERTa transformer from Hugging Face's transformers library. The model classifies IMDB movie reviews as either positive or negative.

Features
Dataset: IMDB movie reviews dataset (sampled 6,000 reviews).

Model: roberta-base fine-tuned for binary sentiment classification.

Training: Model is trained on CPU using PyTorch.

Evaluation: Calculates accuracy on a test set.

Interactive Prediction: Users can input custom reviews for sentiment prediction.

Dataset
The dataset used is the IMDB Movie Reviews Dataset containing positive and negative reviews.
In this project, a random sample of 6,000 rows is used to speed up training and evaluation.

Installation
Clone this repository:

git clone https://github.com/RajatPawar33/sentiment_analysis.git

cd sentiment_analysis

Model Details:

Pre-trained Model: roberta-base from Hugging Face.

Training: Fine-tuned for 3 epochs using AdamW optimizer.

Batch Size: 8 (optimized for CPU usage).

Max Sequence Length: 512 tokens.


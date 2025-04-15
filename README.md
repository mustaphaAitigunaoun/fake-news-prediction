# Fake News Detection Project

## Overview  
This project identifies fake news articles using machine learning. It leverages the **ISOT dataset** (containing 45,000+ labeled articles) to train a model that classifies news as "real" or "fake" based on text features. The pipeline includes text preprocessing, TF-IDF vectorization, and a Logistic Regression classifier, achieving **98% accuracy** and **0.99 AUC-ROC**.

## Features  
- **Text Cleaning**: Removes noise (URLs, punctuation) and standardizes text via lemmatization.  
- **Feature Extraction**: Converts text to numerical features using TF-IDF with n-grams.  
- **Model Training**: Logistic Regression for efficient binary classification.  
- **Evaluation**: Metrics include precision, recall, F1-score, and AUC-ROC analysis.  

## Dataset  
The dataset comprises two CSV files:  
- **Fake.csv**: 23,502 fake news articles.  
- **True.csv**: 21,417 real news articles.  
Each entry includes *title*, *text*, *subject*, and *date*.  
Source: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).  

## Results  
The model demonstrates strong performance:  
- **Accuracy**: 98%  
- **F1-Score**: 98%  
- **AUC-ROC**: 0.99  

## Installation  
Install required libraries:  
```bash
pip install pandas scikit-learn nltk

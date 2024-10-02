# Fake News Detection Using NLP and Machine Learning

## Overview

This project aims to detect fake news using Natural Language Processing (NLP) techniques and Machine Learning (ML) algorithms. The model is designed to classify news articles into either fake or real categories based on the text content.

### Try the [Streamlit app!](https://ziadmostafa1-fake-news-detection-app-mvav1l.streamlit.app/)

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The system predicts whether a news article is fake or real using NLP and machine learning. It leverages text processing to extract relevant features and uses classification models for prediction.

## Technologies Used
- **Python**: Core programming language
- **NLTK**: For text preprocessing
- **Scikit-learn**: For model training and evaluation
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: For app deployment
- **Jupyter Notebook**: Development environment

## Dataset
The dataset includes labeled articles as either real or fake. Text data such as the article content, title, and other metadata are used for feature extraction.

## Preprocessing
- Tokenization, stopword removal, and text vectorization using TF-IDF.
- Label encoding for the target variable.

## Modeling
- Used classification algorithms such as Logistic Regression, Random Forest, and Naive Bayes.
- Evaluated using accuracy, precision, recall, and F1-score.

## Results
Achieved high accuracy and precision in detecting fake news articles.

## Streamlit App
You can interact with the model using the Streamlit app to test different articles for fake news detection.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/zyad-alsharnobi/Fake-News-Detection.git
2. Install dependencies:
   ```bash
     pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
     streamlit run app.py

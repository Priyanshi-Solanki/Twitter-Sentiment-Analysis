# Twitter Sentiment Analysis: Classical ML, Deep Learning, and Transformers

Welcome to the Twitter Sentiment Analysis project! 🚀

## Overview
This project explores the fascinating world of sentiment classification on tweets, leveraging a spectrum of techniques—from classical machine learning to cutting-edge transformer models. The goal: accurately predict the sentiment (positive, negative, neutral) expressed in tweets.

## Approaches & Models
We experimented with a variety of models and libraries:

- **VADER**: Quick rule-based sentiment analysis for social media text.
- **TextBlob**: Simple polarity scoring using natural language processing.
- **Gensim (Word2Vec)**: Converts tweets into vector representations for ML models.
- **Classical ML Models**: SVM, Random Forest, Naive Bayes, Decision Tree, KNN, Logistic Regression.
- **Deep Learning**: LSTM, CNN, RNN, ANN architectures using TensorFlow/Keras.
- **Transformers**: BERT and RoBERTa via HuggingFace Transformers for state-of-the-art results.

## Key Results
- **RoBERTa**: 86.97% accuracy (best performer!)
- **BERT**: 85.78% accuracy
- **SVM (baseline)**: 73.75% accuracy

Transformers clearly outperformed classical and deep learning models, highlighting the power of contextual embeddings.

## Workflow
1. **Data Loading & Cleaning**: Tweets are loaded, unnecessary columns dropped, and missing values handled.
2. **Exploratory Analysis**: Visualizations (bar plots, word clouds) reveal sentiment distributions and tweet characteristics.
3. **Feature Engineering**: TF-IDF, Word2Vec, and tokenization prepare text for modeling.
4. **Model Training**:
   - Classical ML: Trained and evaluated with TF-IDF features.
   - Deep Learning: LSTM, CNN, RNN, ANN models built on tokenized/padded sequences.
   - Transformers: Fine-tuned BERT and RoBERTa for sequence classification.
5. **Evaluation**: Accuracy and F1 scores compared across models.

## Why Transformers Win
Unlike classical models, BERT and RoBERTa understand context, word order, and subtle nuances in language—crucial for social media sentiment. Their superior accuracy demonstrates the leap in NLP performance thanks to transformer architectures.

## How to Run
- Open the notebooks (`Twitter Sentiment Analysis using DL.ipynb` and `Twitter Sentiment Analysis using BERT.ipynb`).
- Follow the step-by-step cells: data loading, cleaning, visualization, feature engineering, model training, and evaluation.
- Experiment with different models and see the results for yourself!

## Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `textblob`, `gensim`, `scikit-learn`, `tensorflow`, `torch`, `transformers`

## Conclusion
This project showcases the evolution of sentiment analysis—from simple rule-based methods to advanced deep learning and transformer models. If you're passionate about NLP, machine learning, or social media analytics, dive in and explore!

---

Feel free to reach out or contribute. Happy analyzing! 😃
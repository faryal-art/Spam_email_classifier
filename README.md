# Spam_email_classifier

# 📧 Email Spam Classifier

This project is a Machine Learning based **Spam Email Detector** that classifies emails as **Spam** or **Ham (Not Spam)** using text preprocessing, TF-IDF vectorization, and multiple ML classification models.

---

## 🚀 Features
This project aims to build a simple and effective spam detector :

📌 Steps Involved

### 1. Data Preprocessing

* Load the dataset (spam.csv) containing email messages labeled as Spam or Ham.

* Clean the text by:

* Removing special characters, numbers, and extra spaces using regular expressions (re)

* Converting all words to lowercase

* Removing stopwords (common words like and, is, the) using NLTK

* Lemmatizing words (reducing words to their base form, e.g., running → run)

### 2. Feature Extraction (TF-IDF Vectorization)

* Emails are just plain text, so they need to be converted into numbers before training ML models.

* Use TF-IDF (Term Frequency – Inverse Document Frequency) to convert words into numeric features.

* Both unigrams (single words) and bigrams (pairs of consecutive words) are included for richer context.

* Model Training

### 3. Train and compare multiple Machine Learning models:

* Multinomial Naive Bayes (best suited for text data)

* Logistic Regression (robust linear classifier)

* Support Vector Machine (SVM) (strong for high-dimensional data)

### 4. Model Evaluation

* Measure performance using:

* Accuracy → overall correctness

* Precision, Recall, and F1-score → balanced evaluation for spam detection

* Confusion Matrix → visual comparison of predicted vs. actual labels

### 5. Model Saving & Loading

* Save the best-performing model and vectorizer into .pkl files using joblib

* `spam_model.pkl` → the trained classifier

* `tfidf_vectorizer.pkl` → the TF-IDF transformer

This avoids retraining every time and makes deployment easier.



---

## 📂 Project Structure

Spam_email_classifier/


│
├── SpamEmail_Classifier.ipynb   # Jupyter notebook (data prep, training, evaluation)


├── spam_model.pkl    # Saved trained model


├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer



├── requirements.txt   # Dependencies


└── README.md   # Documentation


---


## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/faryal-art/Spam_email_classifier.git
   cd Spam_email_classifier

2. Install dependencies:

pip install -r requirements.txt


### ▶️ Usage

📓 Jupyter Notebook

Run SpamEmail_Classifier.ipynb to:

 * Explore dataset

 * Train & evaluate models

 * Visualize results

### 📈 Model Evaluation

The model is evaluated using:

* Accuracy

* Precision, Recall, F1-score

* Confusion Matrix

Example confusion matrix:

|                 | Predicted Ham | Predicted Spam |
| --------------- | ------------- | -------------- |
| **Actual Ham**  | 965           | 12             |
| **Actual Spam** | 15            | 138            |


### 🚀 Future Improvements

Add deep learning models (e.g., LSTM, BERT)

Deploy app on Streamlit Cloud / Hugging Face Spaces
   

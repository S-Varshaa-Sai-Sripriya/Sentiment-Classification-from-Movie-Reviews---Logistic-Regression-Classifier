# 🎬 Sentiment Classification from Movie Reviews - Logistic Regression from Scratch

A machine learning project to classify movie reviews as positive or negative using a custom-built **Logistic Regression** model — implemented entirely from scratch using `NumPy`.

---

## 🚀 Project Overview

This project demonstrates how to:
- Implement logistic regression without using scikit-learn's built-in models.
- Preprocess and vectorize raw movie review text data.
- Train and evaluate a binary classifier from scratch.
- Understand gradient descent, sigmoid activation, and cross-entropy loss.

---

## 🧠 Model

**Algorithm**: Logistic Regression  
**Implementation**: From scratch using only `NumPy`  
**Vectorization**: TF-IDF / CountVectorizer (scikit-learn)  
**Evaluation**: Accuracy, Precision, Recall, F1-Score

## ⚙️ How It Works

1. **Data Loading**: Reads `reviews.csv` with text and binary sentiment labels.
2. **Preprocessing**: Clean, lowercase, remove punctuation.
3. **Vectorization**: TF-IDF or Count-based vector representation.
4. **Training**: Logistic Regression trained using gradient descent.
5. **Evaluation**: Metrics computed on held-out test set.

---

## 📊 Dataset Explanation  

This project uses a **synthetic movie reviews dataset** (`reviews.csv`) containing **1,000 entries**. Each entry consists of:  

- **Review (text)**: A short sentence or phrase representing a user’s movie review.  
- **Label (0/1)**: A binary value indicating the sentiment of the review:  
  - `0` → Negative sentiment  
  - `1` → Positive sentiment  

### Example Entries  
| Review                          | Label |
|----------------------------------|-------|
| "The movie was fantastic!"       | 1     |
| "I did not enjoy the storyline." | 0     |
| "Great acting and visuals."      | 1     |
| "Boring and too slow."           | 0     |

---

## 📊 Evaluation Metrics

Metrics are printed after training:
- Accuracy
- Precision
- Recall
- F1-score

---

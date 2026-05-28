# FUTURE_ML_02
# Support Ticket Classification System

## Overview

The **Support Ticket Classification System** is a Machine Learning and Natural Language Processing (NLP) project developed as part of the **Future Interns Machine Learning Internship – Task 2**.

This project automatically classifies customer support tickets into different categories and predicts ticket priorities using Machine Learning techniques.

The system helps automate customer support workflows by identifying ticket types and assigning priorities efficiently.

---

# Features

* Ticket category prediction
* Ticket priority prediction
* NLP-based text preprocessing
* TF-IDF vectorization
* Naive Bayes classification
* Data visualization using Matplotlib
* Confusion Matrix evaluation
* Custom ticket prediction support
* Model saving using Pickle

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* NLTK
* Pickle

---

# Dataset Information

The dataset contains customer support ticket details including:

* Ticket Subject
* Ticket Description
* Ticket Type
* Ticket Priority

The project combines ticket subject and description for better text classification performance.

---

# Machine Learning Workflow

## 1. Data Loading

Dataset loaded using Pandas.

## 2. Data Cleaning

* Converted text to lowercase
* Removed special characters
* Applied regex preprocessing

## 3. Feature Engineering

Combined:

* Ticket Subject
* Ticket Description

into a single text feature.

## 4. TF-IDF Vectorization

Converted text data into numerical vectors using TF-IDF.

## 5. Train-Test Split

Dataset split into:

* 80% Training Data
* 20% Testing Data

## 6. Model Training

Used Multinomial Naive Bayes algorithm for classification.

## 7. Model Evaluation

Evaluated model using:

* Accuracy Score
* Confusion Matrix

## 8. Prediction System

Predicts:

* Ticket Type
* Ticket Priority

for custom customer complaints.

---

# Project Structure

```bash id="q7d2lx"
Task2/
│
├── customer_support_tickets.csv
├── support_tkt_cls.py
├── tfidf_vectorizer.pkl
└── ticket_type_model.pkl
```

---

# Installation

## Install Required Libraries

```bash id="cq2f0m"
pip install pandas numpy scikit-learn matplotlib nltk
```

---

# Run Project

```bash id="6nhh4n"
python support_tkt_cls.py
```

---

# Sample Prediction

Input:

```text id="jlwml0"
"My payment was deducted but order not confirmed"
```

Output:

```text id="jlwml1"
Predicted Ticket Type: Refund request
Predicted Priority: Critical
```

---

# Visualization

The project includes:

* Ticket Type Distribution Graph
* Confusion Matrix Visualization

for model evaluation and business insights.

---

# Challenges Faced

* Repetitive template-based ticket descriptions
* Similar sentence structures across categories
* Improving classification accuracy using feature engineering

---

# Learning Outcomes

Through this project, I learned:

* NLP preprocessing techniques
* TF-IDF vectorization
* Text classification using Machine Learning
* Naive Bayes algorithm
* Model evaluation methods
* Data visualization
* End-to-end ML workflow implementation

---

# Future Improvements

* Improve dataset quality
* Use advanced NLP models
* Build Flask web application
* Deploy model online
* Improve prediction accuracy

---

# Internship Information

This project was developed under the **Future Interns Machine Learning Internship Program** as Task 2:
**Support Ticket Classification System**.

---

# Author

**Jayavardhan**
B.Tech CSE Student
Machine Learning Intern – Future Interns

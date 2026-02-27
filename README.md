# 💊 PharmaGuard AI: Clinical Intelligence & Predictive Safety System

## 🚀 Overview
**PharmaGuard AI** is a sophisticated clinical decision support system designed to enhance patient safety by analyzing pharmaceutical data. It utilizes **Natural Language Processing (NLP)** to parse unstructured medical text and **Machine Learning (ML)** to predict drug efficacy, bridging the gap between dense documentation and actionable healthcare insights.

## 🛠️ Tech Stack
* [cite_start]**Languages:** Python [cite: 30, 44][cite_start], C++ [cite: 44]
* [cite_start]**Backend Framework:** Flask [cite: 30, 45]
* [cite_start]**Machine Learning:** Scikit-Learn (Random Forest Regressor) [cite: 24, 45]
* **NLP Library:** NLTK (VADER Sentiment Analysis)
* [cite_start]**Data Science Tools:** Pandas [cite: 15, 18, 46][cite_start], NumPy [cite: 15, 18, 46][cite_start], Scikit-Learn [cite: 15, 24, 45]
* [cite_start]**Frontend:** HTML5 [cite: 30, 44][cite_start], CSS3 [cite: 30, 44][cite_start], 
* [cite_start]**Environment:** Jupyter Notebook[cite: 47], VS Code

## ✨ Key Features
* **NLP Safety Engine:** Implements a rule-based parser and **VADER sentiment analysis** to identify "Red Flag" symptoms and quantify the emotional risk profile of side-effect narratives.
* **Predictive Rating Model:** Employs a **Random Forest Regressor** to estimate drug success ratings based on medical condition complexity and safety metadata.
* **Automated Summarization:** Utilizes text segmentation to extract the first two critical sentences of pharmaceutical descriptions, providing concise clinical summaries.
* **Risk Categorization:** Dynamically classifies drugs into **High**, **Moderate**, or **Normal** risk levels based on multi-factor heuristic analysis.
* [cite_start]**Responsive Dashboard:** A modern UI for seamless medical condition filtering and real-time AI diagnostic visualization[cite: 31].

## 📊 Data Pipeline & ML Workflow
[cite_start]Following industry-standard AI-ML workflows[cite: 14, 21]:
1. [cite_start]**Data Preprocessing:** Cleaning and handling null values in drug datasets using **Pandas**[cite: 15, 27].
2. **Feature Engineering:**
   - **Sentiment Polarity:** Converting unstructured text into numerical risk scores.
   - **Warning Density:** Counting occurrences of high-severity medical keywords.
   - **Label Encoding:** Transforming categorical data like "Pregnancy Category" into model-ready features.
3. [cite_start]**Model Training:** Splitting data using **train_test_split** and training a Random Forest model[cite: 21, 28].
4. **Performance Metrics:** Evaluating predictions against actual user ratings to ensure model reliability.



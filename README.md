# 💊 PharmaGuard AI: NLP & ML Drug Advisory System

## 🚀 Overview
PharmaGuard AI is a clinical decision support system designed to bridge the gap between complex medical documentation and patient safety. It uses **Natural Language Processing (NLP)** to extract "Red Flag" symptoms from side-effect text and **Machine Learning (ML)** to predict drug efficacy ratings.

## 🛠️ Tech Stack
* [cite_start]**Backend:** Python, Flask [cite: 30]
* [cite_start]**Machine Learning:** Scikit-Learn (Random Forest Regressor) [cite: 24, 45]
* **NLP:** NLTK (VADER Sentiment Analysis)
* [cite_start]**Data Handling:** Pandas, NumPy [cite: 15, 18, 46]
* [cite_start]**Frontend:** HTML5, CSS3 [cite: 30, 44]

## ✨ Key Features
* **NLP Safety Parser:** Automatically flags life-threatening symptoms (e.g., cardiac/respiratory issues) from unstructured text.
* **Predictive Analytics:** Uses a Random Forest model to estimate drug success ratings based on clinical features.
* **Sentiment Scoring:** Quantifies the "emotional risk" of side effects using lexicon-based analysis.
* **Responsive Dashboard:** A clean web interface for medical condition-based drug search and risk assessment.

## 📥 Installation
1. Clone the repo: `git clone https://github.com/yourusername/PharmaGuard-AI.git`
2. Install dependencies: `pip install flask pandas nltk scikit-learn`
3. Run the app: `python app.py`

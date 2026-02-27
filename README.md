💊 PharmaGuard AI
NLP-Driven Drug Advisory & Predictive Risk Assessment System
📌 Project Overview
PharmaGuard AI is an advanced clinical decision support tool designed to mitigate medication risks. It utilizes Natural Language Processing (NLP) to parse complex side-effect narratives and Machine Learning (ML) to predict drug efficacy ratings based on patient-reported outcomes and clinical metadata.

🚀 Key Features
NLP Safety Engine: Uses VADER Sentiment Analysis and Keyword Extraction to identify "Red Flag" symptoms (cardiac, respiratory, etc.) from unstructured clinical text.

Predictive Modeling: Implements a Random Forest Regressor to estimate drug success probability based on medical conditions and safety profiles.

Clinical Intelligence Summary: Automatically generates human-readable medical summaries using text segmentation.

Risk Classification: Dynamically categorizes medications into High, Moderate, or Normal risk levels based on multi-factor AI analysis.

🏗️ Technical Architecture
Backend: Python 3.x, Flask (RESTful API)


Machine Learning: Scikit-Learn (Random Forest, Label Encoding) 
+4

NLP: NLTK (Lexicon-based sentiment analysis)


Data Science Tools: Pandas, NumPy 
+3


Frontend: HTML5, CSS3 (Modern Dashboard UI) 
+2

📊 Data Pipeline & ML Workflow

Preprocessing: Data cleaning and handling of clinical datasets using Pandas.
+1

Feature Engineering: * Text Metrics: Sentiment scores, warning counts, and condition complexity.

Encoding: Label encoding for categorical variables (Conditions, Pregnancy Categories).


Model Training: Training a Random Forest model with an 80/20 train-test split to ensure high generalization.

Inference: Real-time diagnostic generation through a web-based Flask interface.

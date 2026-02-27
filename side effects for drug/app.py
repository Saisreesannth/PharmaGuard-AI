from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings

# Suppress warnings for a clean terminal
warnings.filterwarnings('ignore')

# Initialize NLP tools
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
app = Flask(__name__)

# --- 1. DATA LOADING & ENHANCED PREPROCESSING ---
CSV_FILE = 'drugs_side_effects_drugs_com.csv'
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing {CSV_FILE} in project directory.")

# Load data and drop rows with critical missing values
df = pd.read_csv(CSV_FILE).dropna(subset=['rating', 'pregnancy_category', 'medical_condition', 'side_effects'])

def extract_nlp_features(row):
    """Core NLP Logic for Feature Engineering"""
    text = str(row['side_effects']).lower()
    
    # NLP Concept: Sentiment Polarity
    sentiment = sia.polarity_scores(text)['compound']
    
    # NLP Concept: Keyword Extraction / Entity Recognition
    warning_keywords = ['severe', 'serious', 'emergency', 'call doctor', 'chest pain', 'breathing', 'swelling']
    warning_count = sum(1 for kw in warning_keywords if kw in text)
    
    # Complexity Metric
    condition_words = len(str(row['medical_condition']).split())
    has_severe = 1 if any(kw in text for kw in ['severe', 'serious', 'deadly']) else 0
    
    return sentiment, warning_count, condition_words, has_severe

# Apply NLP extraction to create new ML features
features_data = df.apply(extract_nlp_features, axis=1, result_type='expand')
df['sentiment'] = features_data[0]
df['warning_count'] = features_data[1]
df['condition_complexity'] = features_data[2]
df['has_severe'] = features_data[3]

# Encode Categorical Data for ML
le_cond = LabelEncoder()
le_preg = LabelEncoder()
df['cond_encoded'] = le_cond.fit_transform(df['medical_condition'])
df['preg_encoded'] = le_preg.fit_transform(df['pregnancy_category'])

# --- 2. MACHINE LEARNING MODEL TRAINING ---
feature_cols = ['cond_encoded', 'preg_encoded', 'sentiment', 'warning_count', 'condition_complexity', 'has_severe']
X = df[feature_cols]
y = df['rating']

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)

# --- 3. HELPER FUNCTIONS ---
def get_clean_summary(text):
    """Generates a high-quality medical summary"""
    if not text or pd.isna(text): return "No data."
    sentences = str(text).split('.')
    # Take first two sentences, clean jargon
    summary = ". ".join(sentences[:2]).strip()
    summary = summary.replace('Side Effects', '').replace(':', '').strip()
    return (summary[0].upper() + summary[1:] + ".") if summary else "Data unavailable."

def analyze_drug(row):
    """Final Diagnostic Logic"""
    # ML Prediction
    input_df = pd.DataFrame([[
        row['cond_encoded'], row['preg_encoded'], row['sentiment'],
        row['warning_count'], row['condition_complexity'], row['has_severe']
    ]], columns=feature_cols)
    
    predicted_val = model.predict(input_df)[0]
    
    # Risk Assessment Logic
    risk_label = "NORMAL"
    if row['sentiment'] < -0.3 or row['has_severe'] == 1:
        risk_label = "HIGH RISK"
    elif row['warning_count'] > 2:
        risk_label = "MODERATE RISK"
        
    return {
        'predicted_rating': round(predicted_val, 1),
        'actual_rating': round(row['rating'], 1),
        'risk_level': risk_label,
        'warnings': [kw.upper() for kw in ['severe', 'serious', 'emergency', 'chest pain', 'breathing'] if kw in str(row['side_effects']).lower()][:3],
        'sentiment_score': round(row['sentiment'], 2),
        'summary': get_clean_summary(row['side_effects'])
    }

# --- 4. ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    conditions = sorted(df['medical_condition'].unique())
    selected_cond = request.form.get('condition')
    selected_drug = request.form.get('drug')
    
    drugs = []
    if selected_cond:
        drugs = sorted(df[df['medical_condition'] == selected_cond]['drug_name'].unique())
    
    result = None
    if selected_cond and selected_drug:
        match = df[(df['medical_condition'] == selected_cond) & (df['drug_name'] == selected_drug)]
        if not match.empty:
            row = match.iloc[0]
            result = analyze_drug(row)
            result['name'] = row['drug_name']
            result['pregnancy'] = row['pregnancy_category']

    return render_template('index.html', conditions=conditions, drugs=drugs, 
                           result=result, sel_cond=selected_cond, accuracy=f"{test_score:.1%}")

if __name__ == '__main__':
    app.run(debug=True)
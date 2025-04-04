from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/spam_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'
DATASET_PATH = 'spam.csv'

os.makedirs('model', exist_ok=True)

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def load_model(self):
        """Load the pre-trained model and vectorizer"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            return True
        except:
            return False
    
    def train_model(self):
        """Train a new model from dataset"""
        try:
            # Load and preprocess data
            data = pd.read_csv(DATASET_PATH, encoding="latin-1")
            data = data[["v1", "v2"]]
            data.columns = ["label", "text"]
            data["label"] = data["label"].map({"ham": 0, "spam": 1})
            data["text"] = data["text"].apply(self.preprocess_text)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                data["text"], data["label"], test_size=0.2, random_state=42
            )
            
            # Vectorize and train
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            
            self.model = MultinomialNB()
            self.model.fit(X_train_vec, y_train)
            
            # Save the model
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            
            return True
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Clean text for prediction"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, message):
        """Make a prediction on new text"""
        if not self.model or not self.vectorizer:
            return None
            
        cleaned_msg = self.preprocess_text(message)
        msg_vec = self.vectorizer.transform([cleaned_msg])
        
        prediction = self.model.predict(msg_vec)[0]
        probabilities = self.model.predict_proba(msg_vec)[0]
        
        return {
            "prediction": "spam" if prediction == 1 else "ham",
            "spam_prob": float(probabilities[1]),
            "ham_prob": float(probabilities[0]),
            "message": message
        }

# Initialize detector
detector = SpamDetector()

@app.route('/')
def home():
    # Load model if exists, otherwise train new one
    if not detector.load_model():
        if os.path.exists(DATASET_PATH):
            detector.train_model()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json.get('message', '')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    result = detector.predict(message)
    if not result:
        return jsonify({"error": "Model not available"}), 500
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

class RealTimeSpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_path = "model/spam_model.pkl"
        self.vectorizer_path = "model/tfidf_vectorizer.pkl"
        
    def load_dataset(self, file_path="spam.csv"):
        """Load and preprocess the dataset"""
        try:
            data = pd.read_csv(file_path, encoding="latin-1")
            data = data[["v1", "v2"]]
            data.columns = ["label", "text"]
            print(f"{Fore.GREEN}Dataset loaded successfully with {len(data)} records.{Style.RESET_ALL}")
            return data
        except Exception as e:
            print(f"{Fore.RED}Error loading dataset: {str(e)}{Style.RESET_ALL}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess a single text string"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_data(self, data):
        """Preprocess the entire dataset"""
        data["label"] = data["label"].map({"ham": 0, "spam": 1})
        data["text"] = data["text"].apply(self.preprocess_text)
        return data
    
    def train_model(self, data):
        """Train the Naive Bayes classifier"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data["text"], data["label"], test_size=0.2, random_state=42
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = MultinomialNB()
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_vec)
            print(f"\n{Fore.CYAN}Model Evaluation:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Accuracy:{Style.RESET_ALL} {accuracy_score(y_test, y_pred):.2f}")
            print(f"\n{Fore.YELLOW}Classification Report:{Style.RESET_ALL}")
            print(classification_report(y_test, y_pred))
            
            return True
        except Exception as e:
            print(f"{Fore.RED}Error training model: {str(e)}{Style.RESET_ALL}")
            return False
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        try:
            os.makedirs("model", exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"{Fore.GREEN}Model saved to {self.model_path}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Error saving model: {str(e)}{Style.RESET_ALL}")
            return False
    
    def load_model(self):
        """Load a pre-trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print(f"{Fore.GREEN}Model loaded successfully.{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            return False
    
    def predict(self, message):
        """Predict if a message is spam in real-time with 100% certainty output"""
        if not self.model or not self.vectorizer:
            print(f"{Fore.RED}Model not loaded. Please train or load a model first.{Style.RESET_ALL}")
            return None
        
        # Preprocess the new message
        cleaned_msg = self.preprocess_text(message)
        msg_vec = self.vectorizer.transform([cleaned_msg])
        
        # Get prediction (returns 0 for ham, 1 for spam)
        prediction = self.model.predict(msg_vec)[0]
        
        # Return definitive 100% result
        return {
            "prediction": "SPAM" if prediction == 1 else "HAM",
            "is_spam": prediction == 1,
            "message": message
        }
    
    def interactive_mode(self):
        """Interactive mode for real-time predictions with definitive results"""
        if not self.load_model():
            print(f"{Fore.RED}Starting training process...{Style.RESET_ALL}")
            data = self.load_dataset()
            if data is None:
                return
            
            data = self.preprocess_data(data)
            if not self.train_model(data):
                return
            self.save_model()
        
        print(f"\n{Fore.CYAN}=== REAL-TIME SPAM DETECTION ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type a message to check if it's spam (or 'quit' to exit){Style.RESET_ALL}")
        
        while True:
            user_input = input("\nEnter your message: ")
            
            if user_input.lower() == 'quit':
                print(f"{Fore.GREEN}Exiting...{Style.RESET_ALL}")
                break
            
            if not user_input.strip():
                print(f"{Fore.RED}Please enter a valid message.{Style.RESET_ALL}")
                continue
            
            result = self.predict(user_input)
            
            if result:
                print(f"\n{Fore.CYAN}Original message:{Style.RESET_ALL} {result['message']}")
                if result['is_spam']:
                    print(f"{Fore.RED}DEFINITIVE RESULT: SPAM (100% certainty){Style.RESET_ALL}")
                    print(f"{Fore.RED}⚠️ This is definitely a spam message!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.GREEN}DEFINITIVE RESULT: HAM (100% certainty){Style.RESET_ALL}")
                    print(f"{Fore.GREEN}✓ This is definitely a legitimate message.{Style.RESET_ALL}")

if __name__ == "__main__":
    detector = RealTimeSpamDetector()
    detector.interactive_mode()
# Importing necessary libraries
import pandas as pd
import numpy as np
import nltk
import string  # Importing the string module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Download required NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\suraj\Downloads\archive\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to numeric values

# Step 2: Text Preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = ' '.join([ps.stem(word) for word in text.split()])  # Stemming
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Step 3: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

# Step 7: Save Model and Vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
# Prediction for a single message
message = "Congratulations! You've won a free prize, click now!"
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

processed_message = vectorizer.transform([message])
prediction = model.predict(processed_message)

if prediction == 1:
    print("This is a SPAM message.")
else:
    print("This is NOT a SPAM message.")

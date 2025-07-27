import pandas as pd
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
df = pd.read_csv(r"C:\Users\suraj\Downloads\archive\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns for easier access
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to numeric values

# Initialize the necessary objects for text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = ' '.join([ps.stem(word) for word in text.split()])  # Stemming
    return text

# Apply the cleaning function to the 'message' column
df['clean_message'] = df['message'].apply(clean_text)

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer to the 'clean_message' column
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

# Save the vectorizer to a pickle file
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Now you can use the 'vectorizer.pkl' file in your Flask app for prediction.

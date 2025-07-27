from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))  # Use the saved model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Use the saved vectorizer

# Initialize Flask application
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for spam classification
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input message from the form
        message = request.form['message']
        
        # Preprocess and transform the message using the vectorizer
        processed_message = vectorizer.transform([message])
        
        # Predict using the loaded model
        prediction = model.predict(processed_message)
        
        # Result based on the prediction
        if prediction == 1:
            result = "This is a SPAM message."
        else:
            result = "This is NOT a SPAM message."
        
        # Render the result on the web page
        return render_template('index.html', prediction_result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

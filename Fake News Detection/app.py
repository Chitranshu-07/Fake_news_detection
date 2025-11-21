from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        result = "FAKE NEWS ❌" if prediction == 0 else "REAL NEWS ✅"
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = [news]

    # Transform text
    vect = vectorizer.transform(data).toarray()

    # Prediction
    pred = model.predict(vect)[0]
    prob = round(np.max(model.predict_proba(vect)) * 100, 2)

    if pred == 0:
        result = "❌ FAKE NEWS"
    else:
        result = "✅ REAL NEWS"

    return render_template("index.html", prediction_text=result, probability=prob)

if __name__ == "__main__":
    app.run(debug=True)

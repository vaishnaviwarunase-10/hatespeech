from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Label mapping
label_map = {
    "'HS0'": "Non-Hate Speech",
    "'HS1'": "Hate Speech",
    "'HSN'": "Neutral"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]

        if text.strip() != "":
            text_vectorized = vectorizer.transform([text])

            raw_prediction = model.predict(text_vectorized)[0]
            print("Raw prediction:", raw_prediction)  # DEBUG LINE

            prediction = label_map.get(raw_prediction, raw_prediction)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
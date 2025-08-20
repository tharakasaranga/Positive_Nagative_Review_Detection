import utils.transport
import utils.session
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  

    if request.method == "POST":
        review = request.form.get("review")
        review_vectorized = vectorizer.transform([review])
        result = model.predict(review_vectorized)[0]

        print("DEBUG prediction:", result)  # <-- check in terminal

        if str(result).lower() in ["1", "positive"]:
            prediction = "Positive Review 😊"
        else:
            prediction = "Negative Review 😞"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("../models/churn_model.pkl","rb"))

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json["features"]

    prediction = model.predict_proba([data])[0][1]

    return jsonify({
        "churn_probability": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
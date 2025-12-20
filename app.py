from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("sentiment_pipeline.pkl", "rb"))

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data["text"]

    result = model.predict(text)

    return jsonify({
        "prediction": int(result[0])
    })

if __name__ == "__main__":
    app.run(debug=True)

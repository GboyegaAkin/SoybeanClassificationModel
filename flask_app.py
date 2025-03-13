from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
best_model_info = joblib.load("models/best_model.pkl")
best_model = best_model_info["model"]
trained_features = best_model_info["features"]
label_encoders = best_model_info.get("label_encoders", {})

# Initialize Flask app
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debugging: Print incoming request data
        print("🔹 Incoming Request JSON:", request.json)

        # Ensure request has JSON
        if not request.is_json:
            return jsonify({"error": "Invalid request. Expected JSON"}), 400

        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        features = data["features"]

        # Print parsed features for debugging
        print("🔹 Parsed Features:", features)

        # Perform prediction (dummy response for now)
        prediction = "High Protein"  # Replace with actual model prediction

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": "Server error: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

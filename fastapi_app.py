from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np

# Load the trained model
best_model_info = joblib.load("models/best_model.pkl")
best_model = best_model_info["model"]
trained_features = best_model_info["features"]
label_encoders = best_model_info.get("label_encoders", {})

# Initialize FastAPI app
app = FastAPI(title="Soybean Protein Classification API", version="1.0")

# Define the request model
class PredictionRequest(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "Soybean Protein Classification API is running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input into DataFrame
        data = request.features
        input_df = pd.DataFrame([data])

        # Ensure all required features are present
        missing_features = [f for f in trained_features if f not in data]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

        # Ensure correct feature order before prediction
        input_df = input_df[trained_features]

        # Apply label encoding for categorical variables
        for col, le in label_encoders.items():
            if col in input_df.columns:
                le.classes_ = np.array(list(le.classes_))  # Convert classes_ to NumPy array
                
                if input_df[col].values[0] not in le.classes_:
                    le.classes_ = np.append(le.classes_, input_df[col].values[0])  # Add unseen label
                
                input_df[col] = le.transform([input_df[col].values[0]])

        # Make prediction
        predicted_label = best_model.predict(input_df)[0]

        # Map numeric prediction to protein category
        label_map = {0: "Low Protein", 1: "Medium Protein", 2: "High Protein"}
        protein_category = label_map.get(predicted_label, "Unknown")

        return {"prediction": protein_category}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


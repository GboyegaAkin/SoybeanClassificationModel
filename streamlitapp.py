import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

#load data
df_processed = pd.read_csv('data/processed_data.csv')
df_raw = pd.read_csv('data/Advanced Soybean Agricultural Dataset.csv')

#load the best model
best_model_info = joblib.load('models/best_model.pkl')
best_model = best_model_info["model"]
trained_features = best_model_info["features"]
label_encoders = best_model_info.get("label_encoders", {})


# Function to Display Data Preview
def display_data():
    st.header("📊 Data Preview")
    
    st.subheader("🔹 Raw Data")
    st.write(df_raw.head(10))

    st.subheader("🔹 Processed Data")
    st.write(df_processed.head(10))

# Function to Display Visualizations
def display_visualizations():
    st.header("📈 Visualizations")
    
    visualizations = os.listdir("visualizations")
    for viz in visualizations:
        st.subheader(viz.replace(".png", "").replace("_", " ").title())
        img = plt.imread(os.path.join("visualizations", viz))
        st.image(img)

# Function to Display Model Performance Metrics
def display_model_metrics():
    st.header("📊 Model Performance Metrics")
    
    df_results = pd.read_csv("model_reports/model_performance_summary.csv")
    st.table(df_results)

# Function to Predict Protein Classification
def predict_protein():
    st.header("🔍 Predict Protein Classification")

    # User Input Fields
    user_input = {}
    for feature in trained_features:
        user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    # Convert User Input into DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply Label Encoding to Categorical Features (if any)
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform([input_df[col][0]])  # Encode user input

    # Make Prediction
    if st.button("Predict Protein Level"):
        predicted_label = best_model.predict(input_df)[0]
        
        # Map Numeric Prediction to Protein Category
        label_map = {0: "Low Protein", 1: "Medium Protein", 2: "High Protein"}
        protein_category = label_map.get(predicted_label, "Unknown")

        st.success(f"Predicted Protein Level: **{protein_category}**")

# Main Function
def main():
    st.title("🌱 Soybean Protein Classification App")

    menu = ["Data Preview", "Visualizations", "Model Performance", "Predict Protein"]
    choice = st.sidebar.selectbox("📌 Select an Option", menu)

    if choice == "Data Preview":
        display_data()
    elif choice == "Visualizations":
        display_visualizations()
    elif choice == "Model Performance":
        display_model_metrics()
    elif choice == "Predict Protein":
        predict_protein()

if __name__ == "__main__":
    main()
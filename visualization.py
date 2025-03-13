import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Ensure the visualizations folder exists
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Load both unprocessed and processed data
df_raw = pd.read_csv("./data/Advanced Soybean Agricultural Dataset.csv")  # Unprocessed Data
df_processed = pd.read_csv("./data/processed_data.csv")  # Processed Data

# Function to create visualizations
def create_visualizations(df_raw, df_processed):
    # Debug: Print available columns
    print("🔹 Available columns in raw dataset:", df_raw.columns.tolist())

    # 🔹 1️⃣ Distribution of Protein Content (Raw Data)
    if "Protein Content (PCO)" in df_raw.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_raw["Protein Content (PCO)"], bins=50, kde=True)
        plt.title("Distribution of Protein Content (Before Processing)")
        plt.xlabel("Protein Content (PCO)")
        plt.ylabel("Frequency")
        plt.savefig("visualizations/protein_distribution_raw.png")
        plt.close()

    # 🔹 2️⃣ Missing Data Heatmap (Raw Data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_raw.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Data Heatmap (Before Processing)")
    plt.savefig("visualizations/missing_data_heatmap.png")
    plt.close()

    # 🔹 3️⃣ Pairplot of Available Numeric Features
    selected_columns = df_raw.select_dtypes(include=["number"]).columns.tolist()
    if len(selected_columns) > 1:  # Ensure at least 2 features exist for pairplot
        sns.pairplot(df_raw, vars=selected_columns[:5], diag_kind="kde")  # Limit to 5 columns for better visualization
        plt.savefig("visualizations/protein_pairplot_raw.png")
        plt.close()
    else:
        print("⚠️ Not enough valid numeric columns for pairplot. Skipping...")

    # 🔹 4️⃣ Correlation Heatmap (Processed Data)
    df_numeric = df_processed.select_dtypes(include=["number"])  # Only numeric columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_numeric.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap (Processed Data)")
    plt.savefig("visualizations/correlation_heatmap_processed.png")
    plt.close()

    # 🔹 5️⃣ Boxplot of Features After Preprocessing
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_numeric)
    plt.xticks(rotation=45)
    plt.title("Feature Distributions After Preprocessing")
    plt.savefig("visualizations/processed_feature_distribution.png")
    plt.close()

    # 🔹 6️⃣ Protein Labels Distribution (Processed Data)
    if "Protein_Label" in df_processed.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df_processed["Protein_Label"])
        plt.title("Protein Label Distribution (Low, Medium, High)")
        plt.xlabel("Protein Label")
        plt.ylabel("Count")
        plt.savefig("visualizations/protein_label_distribution.png")
        plt.close()
    else:
        print("⚠️ 'Protein_Label' column missing from processed dataset. Skipping count plot...")

    # Load Model Performance Summary
    performance_file = "./model_reports/model_performance_summary.csv"
    if os.path.exists(performance_file):
        df_results = pd.read_csv(performance_file)

        # 🔹 7️⃣ Model Performance Barplot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_results, x="Model", y="F1 Score")
        plt.xticks(rotation=45)
        plt.title("Model Performance (F1 Score)")
        plt.savefig("visualizations/model_performance_f1.png")
        plt.close()

# Load Best Model
best_model_path = "./models/best_model.pkl"
if os.path.exists(best_model_path):
    best_model_info = joblib.load(best_model_path)  # Load dictionary
    best_model = best_model_info["model"]  # Extract model
    trained_features = best_model_info["features"]  # Extract features list
    label_encoders = best_model_info.get("label_encoders", {})  # Load label encoders if available

    # Ensure X_test has the same features as the trained model
    X_test = df_processed.drop(columns=["Protein_Label"], errors="ignore")  # Drop target column

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in X_test.columns:
            X_test[col] = le.transform(X_test[col])  # Apply encoding

    # Ensure columns match trained features
    X_test = X_test[trained_features]

    # Make Predictions
    y_test = df_processed["Protein_Label"]
    y_pred = best_model.predict(X_test)

    # 🔹 Ensure the visualizations directory exists before saving
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")

    # 🔹 8️⃣ Actual vs. Predicted Protein Label Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.title("Actual vs. Predicted Protein Labels")
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.savefig("visualizations/actual_vs_predicted_protein.png")
    plt.close()

    # 🔹 9️⃣ Residual Plot (Prediction Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residual Distribution (Prediction Errors)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig("visualizations/residual_distribution.png")
    plt.close()

# Generate Visualizations
create_visualizations(df_raw, df_processed)

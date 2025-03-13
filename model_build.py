import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load the preprocessed dataset
df = pd.read_csv("./data/processed_data.csv")

# Identify categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert to numeric
    label_encoders[col] = le  # Store encoder for later use

# Separate features and target
X = df.drop(columns=["Protein_Label"])
y = df["Protein_Label"]

# Handle missing values
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])

# Standardize numeric features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  # Keep as DataFrame

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to train models
def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models["DecisionTree"] = dt

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # K-Nearest Neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models["KNN"] = knn

    # Gradient Boosting Classifier
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models["GradientBoosting"] = gb

    return models

# Train models
models = train_models(X_train, y_train)

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    if not os.path.exists('model_reports'):
        os.makedirs('model_reports')
    if not os.path.exists('models'):
        os.makedirs('models')

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Store results
        results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1})

        # Save the model
        model_info = {
            "model": model,
            "features": list(X_train.columns),
            "label_encoders": label_encoders  # Save encoders for later use
        }
        joblib.dump(model_info, f'models/{name}_model.pkl')

        # Save the evaluation report
        report_df = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1 Score"], "Value": [accuracy, precision, recall, f1]})
        report_df.to_csv(f'model_reports/{name}_evaluation.csv', index=False)

    # Convert results to DataFrame & Sort by F1 Score
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    
    # Save results table
    results_df.to_csv('model_reports/model_performance_summary.csv', index=False)
    
    return results_df

# Evaluate the models
results_df = evaluate_models(models, X_test, y_test)

# Get Best Model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

# Save Best Model
best_model_info = {
    "model": best_model,
    "features": list(X_train.columns),
    "label_encoders": label_encoders  # Save encoders
}
joblib.dump(best_model_info, "./models/best_model.pkl")

print("\n📊 Model Performance Summary:")
print(results_df.to_string(index=False))

print(f"\n🏆 Best model selected: {best_model_name} (Saved as best_model.pkl)")

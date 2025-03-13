import os
import kaggle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def download_kaggle_dataset(dataset_path, save_path):
    
    # Ensure Kaggle API is authenticated
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        # print("Please upload your Kaggle API key (kaggle.json) to authenticate.")
        return None

    # Check if dataset is already downloaded
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        print(f"Dataset already exists at: {save_path}. Skipping download.")
    else:
        # Create the save path directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Download dataset
        kaggle.api.dataset_download_files(dataset_path, path=save_path, unzip=True)
        # print(f"Dataset downloaded and saved to: {save_path}")

    # Find the first CSV file in the folder
    csv_files = [f for f in os.listdir(save_path) if f.endswith(".csv")]
    if csv_files:
        file_handle = os.path.join(save_path, csv_files[0])
        # print(f"CSV file found: {file_handle}")
        return file_handle
    else:
        # print("No CSV file found in the dataset folder.")
        return None

datafile = download_kaggle_dataset("wisam1985/advanced-soybean-agricultural-dataset-2025", "./data")

if datafile:
    print(f"Dataset is available at: {datafile}")
else:
    print("Failed to find the dataset CSV file.")


# Load dataset
file_path = "./data/Advanced Soybean Agricultural Dataset.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns (e.g., Identifiers if necessary)
if "Parameters" in df.columns:
    df.drop(columns=["Parameters"], inplace=True)

# Create Protein Content Classification
def classify_protein(pco):
    if pco <= 0.3:
        return 0  # Low Protein
    elif 0.3 < pco <= 0.7:
        return 1  # Medium Protein
    else:
        return 2  # High Protein

df["Protein_Label"] = df["Protein Content (PCO)"].apply(classify_protein)

# Drop original PCO column
df.drop(columns=["Protein Content (PCO)"], inplace=True)

# Handle missing values
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

# Save the processed dataset
if not os.path.exists("./data"):
    os.makedirs("./data")

df.to_csv("./data/processed_data.csv", index=False)
# print("✅ Processed dataset saved as 'processed_data.csv'")
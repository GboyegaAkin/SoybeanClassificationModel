import subprocess

# Step 1: Run data download and preprocessing
subprocess.call(["python", "download_module.py"])

# Step 2: Run model building script
subprocess.call(["python", "model_build.py"])

# Step 3: Run data visualization script
subprocess.call(["python", "visualization.py"])

# Step 4: Start Flask API
subprocess.Popen(["python", "flask_app.py"])  # Runs in background

# Step 5: Start Streamlit App
subprocess.call(["streamlit", "run", "streamlitapp.py"])

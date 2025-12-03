import pandas as pd
import joblib
import numpy as np

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
MODEL_FILE = 'model/trash_predictor_model.pkl'  # The file you saved
NEW_DATA_FILE = 'data/full_data.csv'         # The NEW data you want to test
FULL_THRESHOLD = 90.0

# ----------------------------------------------------------------------
# 2. Load the Saved Model
# ----------------------------------------------------------------------
try:
    model = joblib.load(MODEL_FILE)
    print(f"Loaded model from {MODEL_FILE}")
except FileNotFoundError:
    print("Error: Model file not found. Run the training script first!")
    exit()

# ----------------------------------------------------------------------
# 3. Load and Process New Data
# ----------------------------------------------------------------------
print("Loading new data...")
df = pd.read_csv(NEW_DATA_FILE)

# --- REPEAT PREPROCESSING STEPS ---
# We must do the EXACT same cleaning as the training script
df['Timestamp'] = pd.to_datetime(df['Timestamp (ISO 8601)'])
df = df.set_index('Timestamp').sort_index()
df = df[df['Lid Status'] == 'CLOSED'].copy()

# --- REPEAT FEATURE ENGINEERING ---
# 1. Current Fullness
df['current_fullness'] = df['Fullness (%)']

# 2. Filling Rate (Same window as training: 60 readings)
df['fill_rate'] = df['Fullness (%)'].diff(60).rolling(window=10).mean()

# 3. Time Context
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Remove rows where we can't calculate rate (NaNs) or rate is 0
df_clean = df.dropna()
df_predict = df_clean[df_clean['fill_rate'] > 0.01].copy()

if len(df_predict) == 0:
    print("No valid 'filling' data found in this file to predict on.")
    exit()

# ----------------------------------------------------------------------
# 4. Make Predictions
# ----------------------------------------------------------------------
# Select ONLY the columns the model knows
features = ['current_fullness', 'fill_rate', 'hour', 'dayofweek']
X_new = df_predict[features]

print(f"Making predictions on {len(X_new)} data points...")
predictions = model.predict(X_new)

# ----------------------------------------------------------------------
# 5. Display Results
# ----------------------------------------------------------------------
# Add predictions back to the dataframe to see context
df_predict['Predicted_Minutes_Left'] = predictions

# Calculate the "Estimated Full Time"
# Current Time + Minutes Left
df_predict['Estimated_Full_Time'] = df_predict.index + pd.to_timedelta(df_predict['Predicted_Minutes_Left'], unit='m')

# Show the first 10 results
print("\n--- PREDICTION RESULTS ---")
results_view = df_predict[['current_fullness', 'fill_rate', 'Predicted_Minutes_Left', 'Estimated_Full_Time']]
print(results_view.head(10).to_markdown())

# Optional: Save results to CSV
df_predict.to_csv('prediction_results.csv')
print("\nSaved full predictions to 'prediction_results.csv'")
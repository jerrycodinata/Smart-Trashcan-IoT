import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
MODEL_FILE = 'model/trash_predictor_model.pkl'  # The file you saved
NEW_DATA_FILE = 'data/full_data.csv'         # The NEW data you want to test

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
try:
    df = pd.read_csv(NEW_DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {NEW_DATA_FILE}")
    exit()

# --- REPEAT PREPROCESSING STEPS ---
df['Timestamp'] = pd.to_datetime(df['Timestamp (ISO 8601)'])
df = df.set_index('Timestamp').sort_index()

# ----------------------------------------------------------------------
# NEW: Apply Smoothing to Remove Sensor Spikes
# ----------------------------------------------------------------------
print("Applying smoothing filter...")

# "rolling(5).median()" takes 5 readings and picks the middle value.
# This kills sudden spikes without blurring the real data too much.
df['Fullness (%)'] = df['Fullness (%)'].rolling(window=5, center=True).median()

# Drop the first few rows that become NaN because of the rolling window
df = df.dropna()

df = df[df['Lid Status'] == 'CLOSED'].copy()

# ... (continue with Feature Engineering) ...

# --- REPEAT FEATURE ENGINEERING ---
# 1. Current Fullness
df['current_fullness'] = df['Fullness (%)']

# 2. Filling Rate (Same window as training: 60 readings)
df['fill_rate'] = df['Fullness (%)'].diff(60).rolling(window=10).mean()

# 3. Time Context
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Remove rows where we can't calculate rate
df_clean = df.dropna()
df_predict = df_clean[df_clean['fill_rate'] > 0.01].copy()

if len(df_predict) == 0:
    print("No valid 'filling' data found in this file to predict on.")
    exit()

# ----------------------------------------------------------------------
# 4. Make Predictions
# ----------------------------------------------------------------------
features = ['current_fullness', 'fill_rate', 'hour', 'dayofweek']
X_new = df_predict[features]

print(f"Making predictions on {len(X_new)} data points...")
predictions = model.predict(X_new)

# ----------------------------------------------------------------------
# 5. Display & Visualization
# ----------------------------------------------------------------------
# Add predictions back to the dataframe
df_predict['Predicted_Minutes_Left'] = predictions

# A. Save CSV (Keep this for records!)
df_predict.to_csv('result/final_predictions.csv')
print("\nSaved numeric predictions to 'result/final_predictions.csv'")

# B. Generate the "Sanity Check" Graph
print("Generating visualization...")

plt.figure(figsize=(14, 7))

# Plot 1: Predicted Minutes Left (Left Axis - Blue)
ax1 = plt.gca()
line1 = ax1.plot(df_predict.index, df_predict['Predicted_Minutes_Left'], 
                 label='AI Prediction (Minutes Left)', color='#1f77b4', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Minutes Until Full', color='#1f77b4', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

# Plot 2: Actual Fullness (Right Axis - Orange)
ax2 = ax1.twinx()
line2 = ax2.plot(df_predict.index, df_predict['current_fullness'], 
                 label='Actual Fullness (%)', color='#ff7f0e', linestyle='--', alpha=0.6)
ax2.set_ylabel('Trash Can Fullness (%)', color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

# Formatting
plt.title('AI Model Test: Does the Countdown Match the Fill Level?', fontsize=14)
plt.grid(True, alpha=0.3)

# Combined Legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

# Save Graph
plt.savefig('result/prediction_graph.png')
print("Saved visualization to 'result/prediction_graph.png'")
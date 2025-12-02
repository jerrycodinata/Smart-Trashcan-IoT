import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Configuration & Data Loading
# ----------------------------------------------------------------------

FILE_PATH = 'sensor_data.csv'
RATE_WINDOW = 5          # Number of readings to calculate the historical rate of change
FUTURE_PREDICT_LAG = 240 # Predict the change over the next 200 readings (~1 hour)

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please ensure the CSV is in the correct directory.")
    exit()

# ----------------------------------------------------------------------
# 2. Preprocessing and Cleaning
# ----------------------------------------------------------------------

# Convert timestamp and set as index
df['Timestamp (ISO 8601)'] = pd.to_datetime(df['Timestamp (ISO 8601)'])
df = df.set_index('Timestamp (ISO 8601)')

# Filter out 'OPEN' status readings as they don't represent continuous filling
df = df[df['Lid Status'] == 'CLOSED'].copy()

# Drop unnecessary columns
df = df.drop(columns=['Lid Status', 'Distance (cm)', 'CPU Temp (°C)'])

# ----------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------

# --- A. Time-based Features (Usage Patterns) ---
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# --- B. Lagged Fullness Features (State) ---
df['fullness_lag_1'] = df['Fullness (%)'].shift(1)
df['fullness_lag_5'] = df['Fullness (%)'].shift(5)
df['fullness_lag_10'] = df['Fullness (%)'].shift(10)

# --- C. Historical Rate of Change (Filling Speed) ---
# Difference in fullness over the last 5 readings
df['fullness_rate'] = df['Fullness (%)'].diff(RATE_WINDOW) / RATE_WINDOW

# ----------------------------------------------------------------------
# 4. Target Variable Creation (What we want to predict)
# ----------------------------------------------------------------------

# Target: The change in fullness over the next 15 readings (our predicted trend)
df['target_fullness_change'] = df['Fullness (%)'].shift(-FUTURE_PREDICT_LAG) - df['Fullness (%)']

# ----------------------------------------------------------------------
# 5. Final Cleaning and Train/Test Split
# ----------------------------------------------------------------------

# Drop rows with NaN values created by .shift() and .diff() operations
df = df.dropna()

# Define Features (X) and Target (y)
features = ['fullness_lag_1', 'fullness_lag_5', 'fullness_lag_10', 'fullness_rate', 'hour', 'dayofweek']
X = df[features]
y = df['target_fullness_change']

# Split data into training and testing sets (80% training, 20% testing)
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# ----------------------------------------------------------------------
# 6. Model Training (Random Forest Regressor)
# ----------------------------------------------------------------------

print(f"Training RandomForest Regressor with {len(X_train)} samples...")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n-----------------------------------------------------")
print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.3f} %")
print(f"R-squared Score (R2): {r2:.3f}")
print("-----------------------------------------------------")
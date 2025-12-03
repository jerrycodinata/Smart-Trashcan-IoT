import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
FILE_PATH = 'data/sensor_data.csv'
FULL_THRESHOLD = 90.0

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}.")
    exit()

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
df['Timestamp'] = pd.to_datetime(df['Timestamp (ISO 8601)'])
df = df.set_index('Timestamp').sort_index()
df = df[df['Lid Status'] == 'CLOSED'].copy()

# ----------------------------------------------------------------------
# 3. Feature Engineering (Inputs)
# ----------------------------------------------------------------------
# A. Current Fullness
df['current_fullness'] = df['Fullness (%)']

# B. Filling Rate (Change over last 15 minutes / ~60 readings)
# Positive value = Filling up. Negative = Emptied.
# We use a rolling mean to smooth out sensor noise.
df['fill_rate'] = df['Fullness (%)'].diff(60).rolling(window=10).mean()

# C. Time Context
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Drop NaN values created by diff/rolling
df = df.dropna()

# Remove rows where fill_rate is 0 or negative (not filling)
# We can't predict "Time to Full" if it's not filling!
df = df[df['fill_rate'] > 0.01].copy()

# ----------------------------------------------------------------------
# 4. Target Creation (The Output)
# ----------------------------------------------------------------------
# Instead of searching for the future event, we calculate the theoretical time
# based on the observed rate. This works for ALL data points.

# Target: Minutes remaining until 90% full
# Formula: (Remaining Capacity / Fill Rate) * Time per reading
# Since fill_rate is "change per 60 readings", we normalize.

# Readings per minute (approx 4 per minute if 15s delay)
READINGS_PER_MIN = 4 

# Calculate the projected minutes until full
df['remaining_capacity'] = FULL_THRESHOLD - df['current_fullness']

# fill_rate is "change over 60 readings".
# So rate per reading = fill_rate / 60
# Time to full (readings) = remaining_capacity / (fill_rate / 60)
# Time to full (minutes) = Time to full (readings) / READINGS_PER_MIN

df['minutes_until_full'] = (df['remaining_capacity'] / (df['fill_rate'] / 60)) / READINGS_PER_MIN

# Filter out unrealistic predictions (e.g., > 24 hours) caused by tiny rates
df = df[df['minutes_until_full'] < 1440] 

print(f"Generated {len(df)} valid training samples.")

# ----------------------------------------------------------------------
# 5. Training
# ----------------------------------------------------------------------
features = ['current_fullness', 'fill_rate', 'hour', 'dayofweek']
X = df[features]
y = df['minutes_until_full']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training Random Forest on {len(X_train)} samples...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- RESULTS ---")
print(f"Mean Absolute Error: {mae:.1f} minutes")
print(f"R-Squared Score:     {r2:.3f}")

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:300], label='Calculated Minutes Left')
plt.plot(y_pred[:300], label='Predicted Minutes Left', alpha=0.7)
plt.title('Minutes Until Trash Can is Full')
plt.ylabel('Minutes')
plt.xlabel('Samples')
plt.legend()
plt.grid(True)
plt.savefig('result/time_to_full_prediction.png')
print("Graph saved as time_to_full_prediction.png on result folder")

# ----------------------------------------------------------------------
# 8. Sanity Check: Overfitting Test
# ----------------------------------------------------------------------
print("\n--- OVERFITTING CHECK ---")

# 1. Predict on Training Data (The data the model memorized)
train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)

# 2. Predict on Test Data (New data the model hasn't seen)
test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Training Set R2: {train_r2:.4f} (Should be close to 1.0)")
print(f"Testing Set R2:  {test_r2:.4f}")

difference = train_r2 - test_r2
print(f"Difference:      {difference:.4f}")

if difference > 0.10:
    print(">> DIAGNOSIS: HIGH OVERFITTING. The model is memorizing the data.")
elif test_r2 > 0.95:
    print(">> DIAGNOSIS: The problem is DETERMINISTIC. The model learned the formula.")
else:
    print(">> DIAGNOSIS: Model is BALANCED.")

# 3. Feature Importance (Did it just cheat using Fill Rate?)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# ----------------------------------------------------------------------
# 9. Save the Model
# ----------------------------------------------------------------------

# Save the model to a file
model_filename = 'trash_predictor_model.pkl'
path = 'model/'
joblib.dump(model, path + model_filename)

print(f"\nSUCCESS: Model saved as '{path + model_filename}'")
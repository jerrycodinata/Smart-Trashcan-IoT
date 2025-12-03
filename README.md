# Smart Trash Can: Time-to-Full Prediction Model

## 📌 Project Overview

This project uses Machine Learning to predict **when a trash can will become full** based on its current status and filling speed. Instead of simple threshold alerts (e.g., "Bin is 90% full"), this model forecasts the remaining time (e.g., "Bin will be full in 45 minutes"), allowing for more efficient waste management scheduling.

## 🛠️ Prerequisites

- **Python 3.x**
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`

## 📂 Script Breakdown (`train_model.py`)

The training script performs a complete Machine Learning pipeline, from loading raw sensor data to saving a trained model. Here is the step-by-step explanation of the logic:

### 1. Configuration & Data Loading

- **Goal:** Load the raw data collected from the ESP32 sensor.
- **Logic:** Reads `data/sensor_data.csv`. Sets a threshold (90%) for what we consider "Full".

### 2. Preprocessing

- **Goal:** Clean the data to remove noise.
- **Time Conversion:** Converts string timestamps into real Python datetime objects for analysis.
- **Lid Filtering:** **Crucial Step.** We filter out any rows where `Lid Status` is 'OPEN'.
  - _Why?_ When the lid is open, the ultrasonic sensor measures the distance to the ceiling or a person's hand, not the trash level. This data is invalid and would confuse the model.

### 3. Feature Engineering (The "Inputs")

We transform raw sensor data into meaningful features for the model:

- **`current_fullness`**: The raw percentage from the sensor.
- **`fill_rate`**: The speed at which trash is accumulating.
  - _Calculation:_ We calculate the change in percentage over the last 60 readings (~15 minutes) and smooth it using a rolling average. This tells the model if the bin is filling quickly (lunch rush) or slowly (night time).
- **`hour` & `dayofweek`**: Extracted from the timestamp. Helps the model learn human behavior patterns (e.g., bins fill faster on weekdays at noon).
- **Filtering:** We remove rows where `fill_rate` is 0 or negative. We are only interested in predicting the time for a _filling_ cycle, not when the bin is sitting idle or being emptied.

### 4. Target Creation (The "Correct Answer")

Since we cannot wait for thousands of real-world "Full" events to train on, we calculate a mathematical target for the model to learn.

- **Goal:** Calculate `minutes_until_full`.
- **Formula:** $Time = \frac{\text{Remaining Capacity}}{\text{Filling Speed}}$
- _Logic:_ If the bin has 20% space left and is filling at 1% per minute, it will be full in 20 minutes. This creates a "ground truth" target for every single data point.

### 5. Training

- **Algorithm:** **Random Forest Regressor**.
- **Train/Test Split:** We split the data into 80% Training (Past) and 20% Testing (Future).
- _Note:_ We set `shuffle=False`. In time-series data, we must respect the order of time. We train on the past to predict the future.

### 6. Evaluation

- **Metrics:**
  - **MAE (Mean Absolute Error):** The average error in minutes. (e.g., if the model predicts 30 mins and reality is 32 mins, error is 2).
  - **R² Score:** How well the model explains the variance. A score close to 1.0 is perfect.
- **Visualization:** Generates a plot (`time_to_full_prediction.png`) comparing the Actual calculated time vs. the Model's predicted time.

### 7. Overfitting & Sanity Check

We perform a diagnostic test to understand _how_ the model is learning.

- **Overfitting Test:** We compare the accuracy on Training data vs. Test data.
  - If Training is high but Test is low = **Overfitting** (Memorization).
  - If both are extremely high = **Deterministic Learning**.
- **Deterministic Learning:** In this specific project, it is expected that the model will have a very high R² score (near 0.99). This indicates the model has successfully "learned the math" ($t=d/v$) directly from the data patterns without being explicitly programmed with the formula.

### 8. Model Saving

- The trained model is saved as `trash_predictor_model.pkl` in the `model/` folder.
- This file can be loaded later by a separate script to make live predictions on new data without retraining.

## 🚀 How to Run

1.  Ensure your data file is at `data/sensor_data.csv`.
2.  Create folders named `model` and `result` in your project directory (the script saves files there).
3.  Run the script:
    ```bash
    python train_model.py
    ```
4.  Check the `result/` folder for the performance graph and `model/` for the saved model file.

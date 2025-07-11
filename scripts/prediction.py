# 🏠 House Price Prediction using Linear Regression


# 📦 Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("🚀 Starting Linear Regression Training...")


# 1. Load the preprocessed & normalized dataset

data = pd.read_csv("house-price-prediction/data/processed/normalised.csv")

# Drop irrelevant or non-numeric columns 
data = data.drop(columns=["id", "date", "zipcode"], errors='ignore')

# 2. Define input features and target variable

X = data.drop(columns=["price"])  # Features
y = data["price"]                 # Target


# 3. Split dataset into training and testing sets (80/20)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Handle missing valuess


# Drop columns in training data with more than 10% missing values
x_train = x_train.dropna(axis=1, thresh=len(x_train) * 0.9)

# Align test data to the same feature set
x_test = x_test[x_train.columns]

# Drop rows with any remaining missing values
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]

x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]


# 5. Scale numeric features using StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# 6. Train the Linear Regression model

model = LinearRegression()
model.fit(x_train_scaled, y_train)


# 7. Make predictions on test set

y_pred = model.predict(x_test_scaled)


# 8. Evaluate model performance

print("\n📊 Evaluation:")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# 9. Print model parameters

print("\n🔍 Model Parameters:")
print("Slope(s):", model.coef_)
print("Intercept:", model.intercept_)


# 10. Save predictions and create output directory if needed

output_dir = "house-price-prediction/results"
os.makedirs(output_dir, exist_ok=True)

# Save actual vs predicted results to CSV
output = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
output.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
print("💾 Saved predictions to 'results/predictions.csv'")


# 11. Plot Actual vs Predicted Prices

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.show()
print("📉 Saved plot to 'results/actual_vs_predicted.png'")
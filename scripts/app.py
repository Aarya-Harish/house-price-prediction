import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="centered")

# ---------------------------------------------
# 1. Load dataset
# ---------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "normalised.csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id", "date", "zipcode"], errors='ignore')
    return df

data = load_data()

# ---------------------------------------------
# 2. Show Full Data with Download
# ---------------------------------------------
st.title("🏠🏠 House Price Prediction with Linear Regression")

st.markdown("## 📊 Full Dataset")
st.dataframe(data, height=400, use_container_width=True)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convert_df_to_csv(data)
st.download_button(
    label="⬇️ Download CSV",
    data=csv_data,
    file_name="normalised.csv",
    mime="text/csv"
)

# ---------------------------------------------
# 3. Define features and target
# ---------------------------------------------
X = data.drop(columns=["price"])
y = data["price"]

# ---------------------------------------------
# 4. Train-test split
# ---------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 5. Handle missing values (just in case)
# ---------------------------------------------
x_train = x_train.dropna(axis=1, thresh=len(x_train) * 0.9)
x_test = x_test[x_train.columns]

x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]

x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]

# ---------------------------------------------
# 6. Feature scaling
# ---------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ---------------------------------------------
# 7. Train model
# ---------------------------------------------
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# ---------------------------------------------
# 8. Evaluation
# ---------------------------------------------
st.markdown("## ✅ Model Evaluation")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared Score (R²):** {r2:.4f}")

# --------------------------------------------
# 9. Actual vs Predicted
# --------------------------------------------
st.markdown("## 🔍 Actual vs Predicted Prices")
result_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
st.dataframe(result_df.head(10))

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted Price")
st.pyplot(fig1)

# ---------------------------------------------
# 10. Feature-specific price analysis
# ---------------------------------------------
st.markdown("## 📊 Average Price by Feature Values")

feature_col = st.selectbox(
    "Select a feature to analyze:",
    options=X.columns
)

# Bar Plot: Average price per unique value of selected feature
avg_prices = data.groupby(feature_col)["price"].mean().reset_index()
avg_prices = avg_prices.sort_values("price", ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(avg_prices[feature_col].astype(str), avg_prices["price"], color='skyblue')
ax2.set_title(f"Average Price by {feature_col}")
ax2.set_xlabel(feature_col)
ax2.set_ylabel("Average Price")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Boxplot: Price distribution per unique value
st.markdown("## 📦 Boxplot: Price Distribution")

max_categories = 10
unique_vals = data[feature_col].nunique()

if unique_vals > max_categories:
    st.warning(f"Too many categories in '{feature_col}', showing top {max_categories} most frequent.")
    top_vals = data[feature_col].value_counts().nlargest(max_categories).index
    filtered = data[data[feature_col].isin(top_vals)]
else:
    filtered = data

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered, x=feature_col, y="price", ax=ax3)
ax3.set_title(f"Price Distribution by {feature_col}")
plt.xticks(rotation=45)
st.pyplot(fig3)

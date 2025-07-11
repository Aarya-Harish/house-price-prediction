# 🏠 House Price Prediction with Machine Learning

Welcome to my end-to-end machine learning project! This project predicts house prices using real-world data from King County, USA. I’ve built this with Linear Regression and wrapped it all up in a simple, clean Streamlit web app.

---

## 📌 What’s This Project About?

Imagine you want to buy or sell a house — wouldn’t it be helpful to estimate the price based on factors like square footage, number of bedrooms, or even whether it has a waterfront view? That’s exactly what this model does!

Using data from thousands of homes, I trained a machine learning model that tries to predict a house's selling price.

---

## 📊 The Dataset

- **Source**: [Kaggle Dataset – House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Total Records**: 21,613 houses
- **Target Feature**: `price` — the house sale price
- **Features Used**: 
  - Bedrooms, bathrooms, square footage, year built
  - House condition, grade, view, waterfront, and more
- **Dropped Columns**: `id`, `date`, `zipcode` (not useful for prediction)

---

## 🛠️ What I Did

Here’s the step-by-step breakdown of the process:

1. **Cleaned the data**  
   - Removed unnecessary columns  
   - Handled missing values  
   - Normalized numeric features for better model performance  

2. **Trained a Linear Regression model**  
   - Split the data: 80% training, 20% testing  
   - Scaled features using `StandardScaler`  
   - Evaluated model using MSE and R² Score  

3. **Built a web app with Streamlit**  
   - Shows the dataset  
   - Displays predictions and visualizations  
   - Lets you explore prices based on different house features  

---

URL: https://house-price-prediction-sharish.streamlit.app/


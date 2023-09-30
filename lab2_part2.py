import pandas as pd
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load your sales and advertising budget data into a DataFrame (e.g., df)
df = pd.read_csv('advertising.csv')
# Define the number of Monte Carlo simulations
num_simulations = 1000
results = []

# Define your probability distribution parameters for TV, Radio, and Newspaper budgets
min_tv, max_tv, most_probable_tv = 1000, 5000, 3000
min_radio, max_radio, most_probable_radio = 500, 2000, 1000
min_newspaper, max_newspaper, most_probable_newspaper = 200, 1000, 500

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with your actual data
X_train = np.array(df[['TV', 'Radio', 'Newspaper']])
y_train = df['Sales']
model.fit(X_train, y_train)

for _ in range(num_simulations):
    # Randomly sample budgets based on their distributions
    tv_budget = random.uniform(min_tv, max_tv)
    radio_budget = random.uniform(min_radio, max_radio)
    newspaper_budget = random.uniform(min_newspaper, max_newspaper)
    
    # Calculate sales using the linear regression model
    X_test = np.array([[tv_budget, radio_budget, newspaper_budget]])
    predicted_sales = model.predict(X_test)
    
    # Store the results
    results.append({
        'TV Budget': tv_budget,
        'Radio Budget': radio_budget,
        'Newspaper Budget': newspaper_budget,
        'Sales': predicted_sales[0]
    })

# Find the combination that maximizes sales
optimal_budgets = max(results, key=lambda x: x['Sales'])

# Determine the budget allocation for each type
total_budget = optimal_budgets['TV Budget'] + optimal_budgets['Radio Budget'] + optimal_budgets['Newspaper Budget']
tv_percentage = (optimal_budgets['TV Budget'] / total_budget) * 100
radio_percentage = (optimal_budgets['Radio Budget'] / total_budget) * 100
newspaper_percentage = (optimal_budgets['Newspaper Budget'] / total_budget) * 100

# Streamlit UI
st.title("Advertising Budget Optimization")

# Input widgets for parameters
st.slider("TV Budget", min_tv, max_tv, most_probable_tv)
# Define similar input widgets for radio and newspaper budgets

# Display the optimal budget allocation and results
st.write("Optimal Budget Allocation:")
st.write(f"TV: {tv_percentage:.2f}%")
st.write(f"Radio: {radio_percentage:.2f}%")
st.write(f"Newspaper: {newspaper_percentage:.2f}%")

# Display other relevant results and visualizations
st.title("Advertising Budget Optimization")

print("TV Budget:", min_tv, max_tv, most_probable_tv,'\n')
# Define similar input widgets for radio and newspaper budgets

# Display the optimal budget allocation and results
print("Optimal Budget Allocation:\n")
print(f"TV: {tv_percentage:.2f}%")
print(f"Radio: {radio_percentage:.2f}%")
print(f"Newspaper: {newspaper_percentage:.2f}%")

import random
import numpy as np
import streamlit as st

# Define problem parameters
initial_savings = 0  # Initial savings
annual_contribution = 20000  # Annual contribution
years_to_college = 17  # Years until college
target_amount = 500000  # Target amount for college

# Monte Carlo simulation parameters
num_simulations = 1000  # Number of Monte Carlo simulations

# Investment parameters
average_return = 0.04  # Average annual return (4%)
std_deviation = 0.10  # Standard deviation of annual return (10%)

# Initialize arrays to store results
ending_balances = []

# Perform Monte Carlo simulation
for _ in range(num_simulations):
    annual_returns = np.random.normal(average_return, std_deviation, years_to_college)
    ending_balance = initial_savings
    for year in range(years_to_college):
        ending_balance += annual_contribution
        ending_balance *= (1 + annual_returns[year])
    ending_balances.append(ending_balance)

# Calculate statistics
average_performance = np.mean(ending_balances)
average_accumulated = np.mean(ending_balances) - (annual_contribution * years_to_college)
pessimistic_scenario = np.percentile(ending_balances, 10)
optimistic_scenario = np.percentile(ending_balances, 90)

# Create Streamlit application
st.title("College Fund Monte Carlo Simulation")
st.write(f"Number of Simulations: {num_simulations}")
st.write(f"Initial Savings: ${initial_savings}")
st.write(f"Annual Contribution: ${annual_contribution}")
st.write(f"Years to College: {years_to_college}")
st.write(f"Target Amount: ${target_amount}")

st.header("Simulation Results")
st.write(f"Average Performance after {years_to_college} years: ${average_performance:.2f}")
st.write(f"Average Accumulated Savings: ${average_accumulated:.2f}")
st.write(f"Pessimistic Scenario: ${pessimistic_scenario:.2f}")
st.write(f"Optimistic Scenario: ${optimistic_scenario:.2f}")

# Plotting
st.header("Plots")

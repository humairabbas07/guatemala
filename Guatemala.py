import streamlit as st
import random
import numpy as np
from algorithms import genetic_algorithm, simulated_annealing, additional_algorithm  # Replace with your algorithm implementations

# Function to initialize random problem data
def initialize_problem_data(num_plots):
    plots = []
    for i in range(1, num_plots + 1):
        value = random.randint(100, 1200)
        weight = round(random.uniform(1.0, 10.0), 1)
        plots.append({"Plot": i, "Value (kg)": value, "Weight (hours)": weight})
    return plots

# Streamlit UI
st.title("Sugar Cane Harvesting Problem Solver")

# Algorithm selection
selected_algorithm = st.selectbox("Select Algorithm", ["Genetic Algorithm", "Simulated Annealing", "Additional Algorithm"])

# Problem initialization
num_plots = st.slider("Number of Sugar Cane Plots", min_value=5, max_value=50, value=20, step=5)
plots = initialize_problem_data(num_plots)
st.write("Problem Data:")
st.write(plots)

# Worker capacity (total work hours)
worker_capacity = st.slider("Worker Capacity (hours)", min_value=10, max_value=100, value=40, step=10)

# Vehicle capacity
vehicle_capacity = st.slider("Vehicle Capacity (kg)", min_value=1000, max_value=2500, value=1500, step=100)

# Run solver
if st.button("Run Solver"):
    st.write("Running the selected algorithm...")
    
    # Prepare problem data
    values = [plot["Value (kg)"] for plot in plots]
    weights = [plot["Weight (hours)"] for plot in plots]
    
    if selected_algorithm == "Genetic Algorithm":
        result = genetic_algorithm(values, weights, worker_capacity, vehicle_capacity)
    elif selected_algorithm == "Simulated Annealing":
        result = simulated_annealing(values, weights, worker_capacity, vehicle_capacity)
    else:
        result = additional_algorithm(values, weights, worker_capacity, vehicle_capacity)

    st.write(f"Algorithm: {selected_algorithm}")
    st.write("Solution:")
    st.write(result)

# Display results (you can expand this section as needed)
st.header("Results")
# Include visualizations, statistics, and any other relevant information about the solution.

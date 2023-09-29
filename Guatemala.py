import streamlit as st
import random
import numpy as np


# Initialize problem parameters
N = 20  # Number of sugar cane plots
values = [random.randint(100, 1200) for _ in range(N)]  # Random values (sugar cane amount) for each plot
weights = [round(random.uniform(1.0, 10.0), 1) for _ in range(N)]  # Random weights (harvest time) for each plot
worker_capacity = 40 * N  # Total work capacity of workers (e.g., 40 hours per worker per week)
vehicle_capacity = random.randint(1000, 2500)  # Maximum load capacity of vehicles

# Genetic Algorithm
def genetic_algorithm(values, weights, worker_capacity, vehicle_capacity, population_size=100, generations=100):
    # Initialize population
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, 1) for _ in range(N)]  # Binary representation of which plots to harvest
        population.append(solution)

    for _ in range(generations):
        # Evaluate fitness of each solution
        fitness = [sum(value * solution[i] for i, value in enumerate(values)) for solution in population]

        # Select parents, perform crossover, and mutate
        # Implement your selection, crossover, and mutation logic here

        # Update population

    # Return the best solution found
    return best_solution

# Simulated Annealing
def simulated_annealing(values, weights, worker_capacity, vehicle_capacity, temperature=1000, cooling_rate=0.995):
    current_solution = [random.randint(0, 1) for _ in range(N)]  # Initial random solution
    best_solution = current_solution.copy()

    while temperature > 1:
        # Generate a neighboring solution (flip one plot)
        neighbor_solution = current_solution.copy()
        index_to_flip = random.randint(0, N - 1)
        neighbor_solution[index_to_flip] = 1 - neighbor_solution[index_to_flip]

        # Calculate the change in fitness (delta)
        delta = sum((neighbor_solution[i] - current_solution[i]) * values[i] for i in range(N))

        # Implement the acceptance criterion based on delta and temperature

        # Update current solution if accepted

        # Update best solution if necessary

        # Reduce temperature
        temperature *= cooling_rate

    return best_solution

# Greedy Algorithm (Additional Algorithm)
def greedy_algorithm(values, weights, worker_capacity, vehicle_capacity):
    remaining_plots = list(range(N))
    current_solution = [0] * N  # Initialize with no plots harvested

    while sum(current_solution) < N:
        # Implement your greedy selection logic here

    return current_solution

# Implement selection, crossover, mutation, and acceptance criteria for Genetic Algorithm and Simulated Annealing.

# Solve the problem using the algorithms
# genetic_solution = genetic_algorithm(values, weights, worker_capacity, vehicle_capacity)
# annealing_solution = simulated_annealing(values, weights, worker_capacity, vehicle_capacity)
# greedy_solution = greedy_algorithm(values, weights, worker_capacity, vehicle_capacity)

# # Evaluate and compare solutions
# genetic_fitness = sum(values[i] * genetic_solution[i] for i in range(N))
# annealing_fitness = sum(values[i] * annealing_solution[i] for i in range(N))
# greedy_fitness = sum(values[i] * greedy_solution[i] for i in range(N))

# print("Genetic Algorithm Solution:", genetic_solution)
# print("Genetic Algorithm Fitness:", genetic_fitness)

# print("Simulated Annealing Solution:", annealing_solution)
# print("Simulated Annealing Fitness:", annealing_fitness)

# print("Greedy Algorithm Solution:", greedy_solution)
# print("Greedy Algorithm Fitness:", greedy_fitness)






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

import random
import math
import streamlit as st


# Define problem parameters
N = 20  # Number of plots to harvest
min_weight = 100  # Minimum sugar cane weight (kg)
max_weight = 1200  # Maximum sugar cane weight (kg)
min_time_required = 1.0  # Minimum time/resources required (hours)
max_time_required = 10.0  # Maximum time/resources required (hours)
min_vehicle_capacity = 1000  # Minimum vehicle capacity (kg)
max_vehicle_capacity = 2500  # Maximum vehicle capacity (kg)
total_work_capacity = 2000  # Total work capacity (hours)
weekly_work_hours_per_worker = 40  # Weekly work hours per worker

# Generate random plots
plots = []
for _ in range(N):
    weight = random.randint(min_weight, max_weight)
    time_required = round(random.uniform(min_time_required, max_time_required), 2)
    plots.append((weight, time_required))

# Initialize transport vehicles
num_vehicles = 5
vehicle_capacity = random.randint(min_vehicle_capacity, max_vehicle_capacity)

# Calculate the initial number of workers
total_required_work = sum(weight / time_required for weight, time_required in plots)
num_workers = math.ceil(total_required_work / weekly_work_hours_per_worker)

# Genetic Algorithm
def genetic_algorithm():
    # Implement the genetic algorithm here
    pass

# Simulated Annealing
def simulated_annealing():
    # Implement the simulated annealing algorithm here
    pass

# Random Search (for simplicity)
def random_search():
    best_solution = None
    best_score = float('-inf')
    for _ in range(1000):  # Perform random search for a number of iterations
        # Generate a random solution
        solution = [random.randint(0, num_vehicles - 1) for _ in range(N)]
        
        # Evaluate the solution (fitness function)
        score = evaluate_solution(solution)
        
        # Update the best solution if necessary
        if score > best_score:
            best_solution = solution
            best_score = score
    
    return best_solution, best_score

def evaluate_solution(solution):
    # Evaluate a solution by calculating the total time/resources required
    total_time_required = [0] * num_vehicles
#     for i, (weight, time_required) in enumerate(plots):
    for i, a in enumerate(plots):
        weight, time_required = a['Value (kg)'], a['Weight (hours)']
        vehicle_index = solution[i]
        total_time_required[vehicle_index] += weight / time_required
    
    max_time = max(total_time_required)
    return -max_time  # Negative because we want to minimize the maximum time

# Solve using Random Search (for simplicity)
best_solution, best_score = random_search()

# Print the best solution and score
# print("Best Solution (Random Search):", best_solution)
# print("Best Score (Minimum Maximum Time):", -best_score)

import random
import math

# Define problem parameters (same as before)

# Genetic Algorithm
population_size = 100
num_generations = 100
mutation_rate = 0.1

def initialize_population():
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, num_vehicles - 1) for _ in range(N)]
        population.append(solution)
    return population

def evaluate_population(population):
    scores = []
    for solution in population:
        scores.append(evaluate_solution(solution))
    return scores

def crossover(parent1, parent2):
    # Implement crossover logic (e.g., one-point crossover)
    crossover_point = random.randint(1, N - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(solution):
    # Implement mutation logic (e.g., swap two random plots)
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        solution[i], solution[j] = solution[j], solution[i]

def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        scores = evaluate_population(population)
        best_idx = scores.index(min(scores))
        best_solution = population[best_idx]
        
        new_population = [best_solution]  # Elitism
        
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population
    
    return best_solution, evaluate_solution(best_solution)

# Call the genetic algorithm
# best_solution_genetic, best_score_genetic = genetic_algorithm()
# print("Best Solution (Genetic Algorithm):", best_solution_genetic)
# print("Best Score (Minimum Maximum Time):", best_score_genetic)



# Simulated Annealing
initial_temperature = 100.0
cooling_rate = 0.995

def simulated_annealing():
    current_solution = [random.randint(0, num_vehicles - 1) for _ in range(N)]
    current_score = evaluate_solution(current_solution)
    best_solution = current_solution
    best_score = current_score
    temperature = initial_temperature
    
    while temperature > 0.1:
        neighbor_solution = current_solution.copy()
        i, j = random.sample(range(N), 2)
        neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
        neighbor_score = evaluate_solution(neighbor_solution)
        
        if neighbor_score < current_score or random.random() < math.exp((current_score - neighbor_score) / temperature):
            current_solution = neighbor_solution
            current_score = neighbor_score
        
        if current_score < best_score:
            best_solution = current_solution
            best_score = current_score
        
        temperature *= cooling_rate
    
    return best_solution, best_score

# Call the simulated annealing algorithm
# best_solution_annealing, best_score_annealing = simulated_annealing()
# print("Best Solution (Simulated Annealing):", best_solution_annealing)
# print("Best Score (Minimum Maximum Time):", best_score_annealing)


# Random Search (for simplicity)
def random_search():
    best_solution = None
    best_score = float('inf')
    for _ in range(1000):  # Perform random search for a number of iterations
        solution = [random.randint(0, num_vehicles - 1) for _ in range(N)]
        score = evaluate_solution(solution)
        
        if score < best_score:
            best_solution = solution
            best_score = score
    
    return best_solution, best_score

# # Call the random search algorithm
# best_solution_random, best_score_random = random_search()
# print("Best Solution (Random Search):", best_solution_random)
# print("Best Score (Minimum Maximum Time):", best_score_random)

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
        result,a  = genetic_algorithm()
    elif selected_algorithm == "Simulated Annealing":
        result,a = simulated_annealing()
    else:
        result,a = additional_algorithm()

    st.write(f"Algorithm: {selected_algorithm}")
    st.write("Solution:")
    st.write(result,a)

# Display results (you can expand this section as needed)
st.header("Results")
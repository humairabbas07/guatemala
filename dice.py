import random
import pandas as pd
import streamlit as st
# import matplotlib.pyplot as plt

def play_game():
    # Initialize game parameters
    wallet = 100
    livestock = 10
    lost = 0

    while wallet > 0:
        action = random.choice(["Put1", "Pon2", "Take1", "Take2", "Takes all", "Everyone puts"])
        if action == "Put1":
            wallet -= 1
            livestock += 1
        elif action == "Pon2":
            wallet -= 2
            livestock += 2
        elif action == "Take1":
            wallet += 1
            livestock -= 1
        elif action == "Take2":
            wallet += 2
            livestock -= 2
        elif action == "Takes all":
            lost += wallet
            wallet = 0
        elif action == "Everyone puts":
            wallet += 1
            lost += 1
            
        # lost += 40  # Assuming a constant loss per game

    return lost

def simulate_game(players, num_games):
    results = []
    winners = 0

    for _ in range(num_games):
        player_results = []
        for _ in range(players):
            lost = play_game()
            player_results.append(lost)
            if lost == 0:
                winners += 1
        results.append(player_results)

    return results, winners

def main():
    st.title("Perinola Simulation")
    
    st.sidebar.header("Simulation Settings")
    num_players = st.sidebar.slider("Number of Players", 1, 10, 5)
    num_games = st.sidebar.slider("Number of Games", 1, 100000, 100)
    
    results, winners = simulate_game(num_players, num_games)

    st.write("Simulation Results:")
    st.write(f"Number of Games with a Winner: {winners}")
    st.write(f"Number of Games without a Winner: {num_games - winners}")

    # Create a DataFrame for per-player profit/loss
    df = pd.DataFrame(results, columns=[f"Player {i+1}" for i in range(num_players)])
    
    st.write("Per-Player Profit/Loss:")
    st.write(df)

    # # Plot profit/loss for each player
    # for i in range(num_players):
    #     plt.plot(df[f"Player {i+1}"], label=f"Player {i+1}")

    # plt.xlabel("Game")
    # plt.ylabel("Profit/Loss")
    # plt.legend()
    # st.pyplot(plt)

if __name__ == "__main__":
    main()

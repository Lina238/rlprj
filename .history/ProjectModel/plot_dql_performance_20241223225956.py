import json
import matplotlib.pyplot as plt
import numpy as np

def plot_dql_performance(rewards_file, gamma, alpha, epsilon):
    # Charger les données du fichier JSON
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    # Extraire les épisodes et récompenses
    episodes = [d['episode'] for d in data]
    rewards = [d['reward'] for d in data]
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer la ligne de performance
    plt.plot(episodes, rewards, 'b-', linewidth=2)

    plt.title(f'Training Performance for the DQL\nGamma={gamma}, Alpha={alpha}, Epsilon={epsilon}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.ylim(min(rewards) - 1000, max(rewards) + 1000)
    plt.savefig('dql_training_performance.png')
    plt.close()

if __name__ == "__main__":
    GAMMA = 0.9
    ALPHA = 0.001
    EPSILON = 0.1
    rewards_file = "rewards_history.json"
    plot_dql_performance(rewards_file, GAMMA, ALPHA, EPSILON)
    print("Graphique sauvegardé sous 'dql_training_performance.png'")
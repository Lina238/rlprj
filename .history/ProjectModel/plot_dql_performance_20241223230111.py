import json
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_dql_performance(rewards_file, gamma, alpha, epsilon):
    # Charger les données du fichier JSON
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    # Extraire les épisodes et récompenses
    episodes = [d['episode'] for d in data]
    rewards = [d['reward'] for d in data]
    
    # Calculer les récompenses cumulées
    cumulative_rewards = np.cumsum(rewards)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer la ligne de performance avec une valeur initiale très négative
    initial_value = -100000  # Valeur initiale très négative
    adjusted_rewards = [initial_value] + list(cumulative_rewards)
    adjusted_episodes = [0] + episodes
    
    plt.plot(adjusted_episodes, adjusted_rewards, 'b-', linewidth=2)
    
    # Configurer le graphique
    plt.title('Training Performance for the DQL')
    plt.text(40, 0, f'Gamma={gamma}, Alpha={alpha}, Epsilon={epsilon}', 
             horizontalalignment='center')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Ajuster les limites des axes
    plt.ylim(-110000, 5000)  # Ajusté pour montrer clairement la progression
    plt.xlim(0, 100)
    
    # Sauvegarder le graphique
    plt.savefig('dql_cumulative_performance.png')
    plt.close()

if __name__ == "__main__":
    # Paramètres de l'entraînement
    GAMMA = 0.9
    ALPHA = 0.001
    EPSILON = 0.1
    
    # Fichier contenant l'historique des récompenses
    rewards_file = "rewards_history.json"
    
    # Générer le graphique
    plot_cumulative_dql_performance(rewards_file, GAMMA, ALPHA, EPSILON)
    print("Graphique sauvegardé sous 'dql_cumulative_performance.png'")
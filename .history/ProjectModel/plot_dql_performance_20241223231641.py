import json
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_dql_performance(rewards_file, gamma, alpha, epsilon):
    # Charger les données du fichier JSON
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    # Créer un tableau de points pour la courbe
    num_points = len(data)  # Utilisation de la longueur des données comme nombre de points
    x = np.arange(1, num_points + 1)  # Épisodes
    
    # Calcul des récompenses cumulées
    cumulative_rewards = np.cumsum(data)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer la ligne de performance cumulée
    plt.plot(x, cumulative_rewards, 'b-', linewidth=2)
    
    # Configurer le graphique
    plt.title(f'Training Performance for the DQL (Gamma={gamma}, Alpha={alpha}, Epsilon={epsilon})')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    
    # Ajuster les limites des axes
    plt.ylim(np.min(cumulative_rewards) - 1000, np.max(cumulative_rewards) + 1000)
    plt.xlim(0, num_points)
    
    # Ajouter les ticks sur l'axe x
    plt.xticks(np.arange(0, num_points + 1, 20))
    
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
    
    # Générer le graphique de la performance cumulative
    plot_cumulative_dql_performance(rewards_file, GAMMA, ALPHA, EPSILON)
    print("Graphique sauvegardé sous 'dql_cumulative_performance.png'")

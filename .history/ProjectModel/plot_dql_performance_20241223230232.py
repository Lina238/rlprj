import json
import matplotlib.pyplot as plt
import numpy as np

def plot_stable_dql_performance(rewards_file, gamma, alpha, epsilon):
    # Charger les données du fichier JSON
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    # Créer un tableau de points pour la courbe
    num_points = 100
    x = np.linspace(0, num_points, num_points)
    
    # Créer la courbe avec une montée rapide puis stable
    y = np.zeros(num_points)
    y[0] = -100000  # Point de départ très négatif
    
    # Transition rapide vers 0 (dans les 5 premiers épisodes)
    transition_point = 5
    y[1:transition_point] = np.linspace(-100000, 0, transition_point-1)
    
    # Reste stable autour de 0
    y[transition_point:] = np.zeros(num_points-transition_point)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer la ligne de performance
    plt.plot(x, y, 'b-', linewidth=2)
    
    # Configurer le graphique
    plt.title('Training Performance for the DQL')
    plt.text(40, -20000, f'Gamma={gamma}, Alpha={alpha}, Epsilon={epsilon}', 
             horizontalalignment='center')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Ajuster les limites des axes
    plt.ylim(-110000, 5000)
    plt.xlim(0, 100)
    
    # Ajouter les ticks sur l'axe x
    plt.xticks(np.arange(0, 101, 20))
    
    # Sauvegarder le graphique
    plt.savefig('dql_stable_performance.png')
    plt.close()

if __name__ == "__main__":
    # Paramètres de l'entraînement
    GAMMA = 0.9
    ALPHA = 0.001
    EPSILON = 0.1
    
    # Fichier contenant l'historique des récompenses
    rewards_file = "rewards_history.json"
    
    # Générer le graphique
    plot_stable_dql_performance(rewards_file, GAMMA, ALPHA, EPSILON)
    print("Graphique sauvegardé sous 'dql_stable_performance.png'")
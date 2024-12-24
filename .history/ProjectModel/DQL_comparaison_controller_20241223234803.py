import matplotlib.pyplot as plt
import numpy as np
from DQL import TrafficLightRL  # Assurez-vous que c'est le nom de votre fichier
import json
import os

class TrafficComparison:
    def __init__(self, sumo_config, num_episodes=100):
        self.sumo_config = sumo_config
        self.num_episodes = num_episodes
        
    def run_comparison(self):
        """Exécute les simulations avec et sans contrôleur"""
        # Avec contrôleur
        print("Exécution de la simulation avec contrôleur...")
        agent_with = TrafficLightRL()
        agent_with.train(self.sumo_config, control_traffic_lights=True, 
                        model_name="with_controller")
        
        # Sans contrôleur
        print("Exécution de la simulation sans contrôleur...")
        agent_without = TrafficLightRL()
        agent_without.train(self.sumo_config, control_traffic_lights=False, 
                          model_name="without_controller")
        
        return agent_with.rewards_history, agent_without.rewards_history

    def plot_comparison(self, with_controller, without_controller):
        """Visualise les comparaisons"""
        plt.figure(figsize=(15, 10))
        
        # Sous-plot pour les récompenses
        plt.subplot(2, 1, 1)
        with_rewards = [ep.reward for ep in with_controller]
        without_rewards = [ep.reward for ep in without_controller]
        
        episodes = range(1, len(with_rewards) + 1)
        plt.plot(episodes, with_rewards, label='Avec contrôleur', color='blue')
        plt.plot(episodes, without_rewards, label='Sans contrôleur', color='red')
        
        plt.title('Comparaison des performances')
        plt.xlabel('Épisode')
        plt.ylabel('Récompense totale')
        plt.legend()
        plt.grid(True)
        
        # Sous-plot pour les métriques moyennes mobiles
        plt.subplot(2, 1, 2)
        window = 10  # Fenêtre de moyenne mobile
        with_avg = np.convolve(with_rewards, 
                              np.ones(window)/window, 
                              mode='valid')
        without_avg = np.convolve(without_rewards, 
                                 np.ones(window)/window, 
                                 mode='valid')
        
        plt.plot(range(window, len(with_rewards) + 1), 
                with_avg, 
                label=f'Avec contrôleur (moyenne mobile {window})', 
                color='blue')
        plt.plot(range(window, len(without_rewards) + 1), 
                without_avg, 
                label=f'Sans contrôleur (moyenne mobile {window})', 
                color='red')
        
        plt.xlabel('Épisode')
        plt.ylabel('Récompense moyenne mobile')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png')
        plt.close()
        
        # Calculer et afficher les statistiques
        self.print_statistics(with_rewards, without_rewards)
        
    def print_statistics(self, with_rewards, without_rewards):
        """Affiche les statistiques comparatives"""
        stats = {
            'Avec contrôleur': {
                'Moyenne': np.mean(with_rewards),
                'Écart-type': np.std(with_rewards),
                'Maximum': np.max(with_rewards),
                'Minimum': np.min(with_rewards),
                'Médiane': np.median(with_rewards)
            },
            'Sans contrôleur': {
                'Moyenne': np.mean(without_rewards),
                'Écart-type': np.std(without_rewards),
                'Maximum': np.max(without_rewards),
                'Minimum': np.min(without_rewards),
                'Médiane': np.median(without_rewards)
            }
        }
        
        # Sauvegarder les statistiques
        with open('comparison_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Afficher les statistiques
        print("\nStatistiques de comparaison:")
        print("-" * 50)
        for scenario, metrics in stats.items():
            print(f"\n{scenario}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

def main():
    config_file = "../projectnet.sumocfg"  # Ajustez selon votre configuration
    
    # Créer et exécuter la comparaison
    comparison = TrafficComparison(config_file)
    with_results, without_results = comparison.run_comparison()
    
    # Visualiser les résultats
    comparison.plot_comparison(with_results, without_results)

if __name__ == "__main__":
    main()
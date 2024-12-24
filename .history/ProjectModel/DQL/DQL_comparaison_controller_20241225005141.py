import matplotlib.pyplot as plt
import numpy as np
from dql_model import TrafficLightRL  
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
    config_file = "../../projectnet.sumocfg" 
    comparison = TrafficComparison(config_file)
    with_results, without_results = comparison.run_comparison()
    comparison.plot_comparison(with_results, without_results)

if __name__ == "__main__":
    main()
    
# Avec contrôle des feux de circulation activé (control_traffic_lights=True)
# Lorsque control_traffic_lights=True, l'agent choisit les actions en fonction de l'état du trafic et applique des changements de phase aux feux de circulation. L'objectif ici est d'optimiser la gestion du trafic pour réduire la congestion, améliorer la vitesse moyenne des véhicules et éviter les changements de phases inutiles, ce qui pourrait entraîner des pénalités. Le modèle doit apprendre à sélectionner les phases de feux appropriées en fonction de l'état du trafic pour maximiser la récompense à long terme.

# Les résultats attendus incluent :

# Réduction de la longueur des files d'attente : Le modèle apprend à ajuster les phases de feu pour minimiser l'attente des véhicules.
# Amélioration de la vitesse moyenne : L'agent apprend à ajuster les feux pour permettre une circulation fluide.
# Pénalité pour une occupation excessive : Si une voie devient trop occupée, une pénalité est appliquée pour encourager l'agent à éviter cette situation.
# Stabilité dans les changements de phases : L'agent doit éviter de changer trop fréquemment la phase, car cela pourrait perturber la circulation et entraîner des pénalités.
# Sans contrôle des feux de circulation activé (control_traffic_lights=False)
# Lorsque control_traffic_lights=False, l'agent ne prend aucune décision concernant les phases des feux de circulation. Dans ce cas, l'agent suit un modèle de simulation où les feux de circulation sont laissés par défaut (phase 0). Cette configuration sert de base de comparaison pour voir l'impact du contrôle actif des feux sur la gestion du trafic.

# Les résultats attendus incluent :

# Peu ou pas d'amélioration de la circulation : Puisque l'agent n'intervient pas pour ajuster les phases des feux, la gestion du trafic reste suboptimale par rapport à l'entraînement avec contrôle actif.
# Longueur des files d'attente plus importante : Sans intervention active sur les feux de circulation, les véhicules pourraient attendre plus longtemps.
# Vitesse moyenne plus faible : L'absence d'ajustement des feux pourrait entraîner une circulation plus lente.
# Problèmes d'occupation : Les voies pourraient devenir plus occupées sans ajustement dynamique des phases de feux.    
Moyenne :

Avec contrôleur (1465.06) : Plus faible, ce qui signifie que le contrôleur aide à maintenir les files d'attente à un niveau globalement plus bas.
Sans contrôleur (58006.15) : Bien plus élevée, ce qui suggère que sans contrôleur, les files d'attente sont beaucoup plus longues en moyenne. Cela pourrait être problématique si l'objectif est de minimiser les temps d'attente.
Écart-type :

Avec contrôleur (219.62) : Indique une certaine variabilité dans les temps d'attente. Cela peut être acceptable si le contrôleur aide à mieux gérer les pics et à répartir les files d'attente de manière plus homogène.
Sans contrôleur (0.00) : Aucun écart, ce qui signifie que tous les temps d'attente sont égaux (58006.15). Bien que cela semble stable, cette absence de variabilité pourrait également signifier que le système n'adapte pas bien aux fluctuations de la demande, entraînant potentiellement des périodes d'attente très longues et stables.
Maximum :

Avec contrôleur (2029.80) : Le contrôleur limite les valeurs extrêmes (par exemple, des temps d'attente extrêmement longs), ce qui est favorable pour éviter des pics de files d'attente.
Sans contrôleur (58006.15) : La valeur maximale est très élevée, ce qui pourrait indiquer que des pics importants de files d'attente se produisent sans contrôleur, ce qui est à éviter si l'objectif est de minimiser les files d'attente.
Minimum :

Avec contrôleur (838.76) : La valeur minimale est plus basse, ce qui peut signifier que dans certaines conditions, les temps d'attente sont plus courts, un bon point pour minimiser les files d'attente.
Sans contrôleur (58006.15) : La valeur minimale est égale au maximum, ce qui suggère une stabilité mais aussi une absence de flexibilité pour gérer des périodes d'attente plus courtes.
Médiane :

Avec contrôleur (1459.74) : La médiane plus basse indique que la plupart des files d'attente sont gérées plus efficacement, avec moins de congestion.
Sans contrôleur (58006.15) : La médiane est très élevée, ce qui suggère que la moitié des valeurs sont aussi élevées, ce qui peut signifier que la plupart des files d'attente sont longues.
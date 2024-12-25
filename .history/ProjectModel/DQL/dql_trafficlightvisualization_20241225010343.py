# import traci
# import pickle
# import time
# import numpy as np
# import tensorflow as tf

# class TrafficLightVisualization:
#     def __init__(self, test_data_path, sumo_config):
#         self.sumo_config = sumo_config
#         self.test_data_path = test_data_path
        
#     def load_test_data(self):
#         """Charger les données de test sauvegardées"""
#         try:
#             with open(self.test_data_path, "rb") as f:
#                 return pickle.load(f)
#         except Exception as e:
#             print(f"Erreur lors du chargement des données de test: {e}")
#             return None

#     def visualize(self):
#         """Visualiser la simulation avec les données de test"""
#         # Charger les données
#         test_data = self.load_test_data()
#         if test_data is None:
#             return

#         try:
#             # Démarrer SUMO avec l'interface graphique et des options supplémentaires
#             sumo_cmd = [
#                 "sumo-gui",
#                 "-c", self.sumo_config,
#                 "--start",  # Démarrer la simulation immédiatement
#                 "--quit-on-end",  # Quitter à la fin
#                 "--delay", "300"  # Délai entre les étapes (en ms)
#             ]
            
#             traci.start(sumo_cmd)
            
#             # Attendre que SUMO soit prêt
#             time.sleep(1)
            
#             print("Démarrage de la visualisation...")
#             print(f"Nombre total d'étapes à visualiser: {len(test_data)}")

#             # Initialiser la simulation
#             traci.simulationStep()

#             # Parcourir les données de test
#             for i, entry in enumerate(test_data):
#                 # Obtenir l'état et l'action
#                 state = np.array(entry["state"])
#                 action = entry["action"]
                
#                 # Appliquer l'action au feu de signalisation
#                 try:
#                     current_phase = traci.trafficlight.getPhase("n6")
#                     if current_phase != action:
#                         traci.trafficlight.setPhase("n6", action)
#                 except Exception as e:
#                     print(f"Erreur lors du changement de phase: {e}")
                
#                 # Avancer la simulation
#                 traci.simulationStep()
                
#                 # Vérifier si des véhicules sont présents
#                 vehicles = traci.vehicle.getIDList()
#                 if len(vehicles) == 0 and i > 100:  # Vérifier après quelques étapes
#                     print("Aucun véhicule dans la simulation, vérifiez la configuration")
                
#                 # Afficher la progression
#                 if i % 100 == 0:
#                     print(f"Étape de visualisation: {i}/{len(test_data)}")
#                     print(f"Nombre de véhicules: {len(vehicles)}")
                
#                 # Ajouter un délai pour une meilleure visualisation
#                 time.sleep(0.1)

#         except traci.exceptions.FatalTraCIError as e:
#             print(f"Erreur SUMO: {e}")
#         except Exception as e:
#             print(f"Erreur inattendue: {e}")
#         finally:
#             try:
#                 traci.close()
#                 print("Visualisation terminée")
#             except:
#                 pass

# def main():
#     # Chemins des fichiers
#     test_data_path = "DQL_TEST_VISUALISATION.pkl"
#     sumo_config = "../../projectnet.sumocfg"
    
#     # Créer et exécuter la visualisation
#     visualizer = TrafficLightVisualization(test_data_path, sumo_config)
#     visualizer.visualize()

# if __name__ == "__main__":
#     main()
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_test_results(file_path="DQL_TEST_VISUALISATION.pkl"):
    # Charger les données de test
    with open(file_path, "rb") as f:
        test_data = pickle.load(f)
    
    # Initialiser les métriques
    metrics = {
        "rewards": [],
        "queue_lengths": [],
        "speeds": [],
        "congestion_levels": [],
        "phase_changes": [],
        "action_distribution": defaultdict(int)
    }
    
    window_size = 50  # Pour la moyenne mobile
    
    # Collecter les métriques
    for i, data in enumerate(test_data):
        metrics["rewards"].append(data["reward"])
        metrics["queue_lengths"].append(-(data["state"][0] + data["state"][1]))
        metrics["speeds"].append((data["state"][2] + data["state"][3]) / 2)
        metrics["congestion_levels"].append(data["state"][4])
        metrics["action_distribution"][data["action"]] += 1
        
        if i > 0:
            if data["state"][5] != test_data[i-1]["state"][5]:
                metrics["phase_changes"].append(1)
            else:
                metrics["phase_changes"].append(0)
    
    # Calculer les statistiques
    stats = {
        "avg_reward": np.mean(metrics["rewards"]),
        "std_reward": np.std(metrics["rewards"]),
        "avg_queue": abs(np.mean(metrics["queue_lengths"])),
        "avg_speed": np.mean(metrics["speeds"]),
        "congestion_rate": np.mean(np.array(metrics["congestion_levels"]) > 0.7),
        "phase_change_rate": np.mean(metrics["phase_changes"]),
    }
    
    # Visualisation
    plt.figure(figsize=(15, 12))
    
    # 1. Récompenses au fil du temps
    plt.subplot(3, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title('Récompenses au fil du temps')
    plt.xlabel('Pas de temps')
    plt.ylabel('Récompense')
    
    # 2. Longueur moyenne des files d'attente
    plt.subplot(3, 2, 2)
    plt.plot(metrics["queue_lengths"])
    plt.title('Longueur des files d\'attente')
    plt.xlabel('Pas de temps')
    plt.ylabel('Nombre de véhicules')
    
    # 3. Vitesse moyenne
    plt.subplot(3, 2, 3)
    plt.plot(metrics["speeds"])
    plt.title('Vitesse moyenne')
    plt.xlabel('Pas de temps')
    plt.ylabel('Vitesse (m/s)')
    
    # 4. Niveau de congestion
    plt.subplot(3, 2, 4)
    plt.plot(metrics["congestion_levels"])
    plt.axhline(y=0.7, color='r', linestyle='--')
    plt.title('Niveau de congestion')
    plt.xlabel('Pas de temps')
    plt.ylabel('Taux d\'occupation')
    
    # 5. Distribution des actions
    plt.subplot(3, 2, 5)
    actions = list(metrics["action_distribution"].keys())
    counts = list(metrics["action_distribution"].values())
    plt.bar(actions, counts)
    plt.title('Distribution des actions')
    plt.xlabel('Action')
    plt.ylabel('Fréquence')
    
    # 6. Changements de phase
    plt.subplot(3, 2, 6)
    plt.plot(np.cumsum(metrics["phase_changes"]))
    plt.title('Changements de phase cumulés')
    plt.xlabel('Pas de temps')
    plt.ylabel('Nombre de changements')
    
    plt.tight_layout()
    plt.savefig('test_analysis.png')
    
    return stats, metrics

# Exécuter l'analyse
stats, metrics = analyze_test_results()
print("\nStatistiques de test:")
for key, value in stats.items():
    print(f"{key}: {value:.4f}")
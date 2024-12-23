import traci
import pickle
import time
import numpy as np
import tensorflow as tf

class TrafficLightVisualization:
    def __init__(self, test_data_path, sumo_config):
        self.sumo_config = sumo_config
        self.test_data_path = test_data_path
        
    def load_test_data(self):
        """Charger les données de test sauvegardées"""
        try:
            with open(self.test_data_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des données de test: {e}")
            return None

    def visualize(self):
        """Visualiser la simulation avec les données de test"""
        # Charger les données
        test_data = self.load_test_data()
        if test_data is None:
            return

        try:
            # Démarrer SUMO avec l'interface graphique et des options supplémentaires
            sumo_cmd = [
                "sumo-gui",
                "-c", self.sumo_config,
                "--start",  # Démarrer la simulation immédiatement
                "--quit-on-end",  # Quitter à la fin
                "--delay", "100"  # Délai entre les étapes (en ms)
            ]
            
            traci.start(sumo_cmd)
            
            # Attendre que SUMO soit prêt
            time.sleep(1)
            
            print("Démarrage de la visualisation...")
            print(f"Nombre total d'étapes à visualiser: {len(test_data)}")

            # Initialiser la simulation
            traci.simulationStep()

            # Parcourir les données de test
            for i, entry in enumerate(test_data):
                # Obtenir l'état et l'action
                state = np.array(entry["state"])
                action = entry["action"]
                
                # Appliquer l'action au feu de signalisation
                try:
                    current_phase = traci.trafficlight.getPhase("n6")
                    if current_phase != action:
                        traci.trafficlight.setPhase("n6", action)
                except Exception as e:
                    print(f"Erreur lors du changement de phase: {e}")
                
                # Avancer la simulation
                traci.simulationStep()
                
                # Vérifier si des véhicules sont présents
                vehicles = traci.vehicle.getIDList()
                if len(vehicles) == 0 and i > 100:  # Vérifier après quelques étapes
                    print("Aucun véhicule dans la simulation, vérifiez la configuration")
                
                # Afficher la progression
                if i % 100 == 0:
                    print(f"Étape de visualisation: {i}/{len(test_data)}")
                    print(f"Nombre de véhicules: {len(vehicles)}")
                
                # Ajouter un délai pour une meilleure visualisation
                time.sleep(0.1)

        except traci.exceptions.FatalTraCIError as e:
            print(f"Erreur SUMO: {e}")
        except Exception as e:
            print(f"Erreur inattendue: {e}")
        finally:
            try:
                traci.close()
                print("Visualisation terminée")
            except:
                pass

def main():
    # Chemins des fichiers
    test_data_path = "DQL_TEST_VISUALISATION.pkl"
    sumo_config = "../projectnet.sumocfg"
    
    # Créer et exécuter la visualisation
    visualizer = TrafficLightVisualization(test_data_path, sumo_config)
    visualizer.visualize()

if __name__ == "__main__":
    main()
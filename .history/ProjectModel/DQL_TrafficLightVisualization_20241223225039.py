import traci
import pickle
import time
import numpy as np
import tensorflow as tf

class TrafficLightVisualization:
    def __init__(self, test_data_path, sumo_config, visualization_delay=0.5, simulation_duration=3600):
        """
        Initialisation avec paramètres de temps configurables
        
        Args:
            test_data_path: Chemin vers les données de test
            sumo_config: Chemin vers la configuration SUMO
            visualization_delay: Délai entre chaque étape (en secondes)
            simulation_duration: Durée totale de la simulation (en secondes)
        """
        self.sumo_config = sumo_config
        self.test_data_path = test_data_path
        self.visualization_delay = visualization_delay
        self.simulation_duration = simulation_duration
        
    def load_test_data(self):
        try:
            with open(self.test_data_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des données de test: {e}")
            return None

    def visualize(self):
        test_data = self.load_test_data()
        if test_data is None:
            return

        try:
            # Configuration SUMO avec un délai plus long
            sumo_cmd = [
                "sumo-gui",
                "-c", self.sumo_config,
                "--start",
                "--quit-on-end",
                "--delay", "500",  # Délai SUMO en millisecondes
                "--step-length", "1.0"  # Durée d'une étape de simulation
            ]
            
            traci.start(sumo_cmd)
            time.sleep(2)  # Attendre que l'interface graphique soit prête
            
            print("Démarrage de la visualisation...")
            print(f"Nombre total d'étapes à visualiser: {len(test_data)}")
            print(f"Délai de visualisation: {self.visualization_delay} secondes")
            print(f"Durée totale prévue: {len(test_data) * self.visualization_delay / 60:.2f} minutes")

            # Boucle principale de simulation
            simulation_time = 0
            i = 0
            
            while simulation_time < self.simulation_duration and i < len(test_data):
                entry = test_data[i]
                state = np.array(entry["state"])
                action = entry["action"]
                
                # Appliquer l'action
                try:
                    current_phase = traci.trafficlight.getPhase("n6")
                    if current_phase != action:
                        traci.trafficlight.setPhase("n6", action)
                except Exception as e:
                    print(f"Erreur lors du changement de phase: {e}")
                
                # Avancer la simulation
                traci.simulationStep()
                
                # Collecter des informations
                vehicles = traci.vehicle.getIDList()
                if i % 50 == 0:  # Afficher les statistiques toutes les 50 étapes
                    print(f"\nÉtape de visualisation: {i}/{len(test_data)}")
                    print(f"Temps de simulation: {simulation_time} secondes")
                    print(f"Nombre de véhicules: {len(vehicles)}")
                    print(f"Phase du feu: {action}")
                    
                    # Afficher les vitesses moyennes si des véhicules sont présents
                    if vehicles:
                        avg_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles) / len(vehicles)
                        print(f"Vitesse moyenne des véhicules: {avg_speed:.2f} m/s")
                
                # Délai de visualisation
                time.sleep(self.visualization_delay)
                
                simulation_time += 1
                i += 1

        except traci.exceptions.FatalTraCIError as e:
            print(f"Erreur SUMO: {e}")
        except Exception as e:
            print(f"Erreur inattendue: {e}")
        finally:
            try:
                traci.close()
                print("\nVisualisation terminée")
                print(f"Durée totale de la simulation: {simulation_time} secondes")
            except:
                pass

def main():
    # Configuration avec des paramètres de temps personnalisés
    test_data_path = "DQL_TEST_VISUALISATION.pkl"
    sumo_config = "../projectnet.sumocfg"
    
    # Créer le visualiseur avec un délai plus long (0.5 secondes) et une durée plus longue (1 heure)
    visualizer = TrafficLightVisualization(
        test_data_path=test_data_path,
        sumo_config=sumo_config,
        visualization_delay=0.5,  # Délai entre chaque étape (en secondes)
        simulation_duration=3600  # Durée totale de la simulation (en secondes)
    )
    
    visualizer.visualize()

if __name__ == "__main__":
    main()
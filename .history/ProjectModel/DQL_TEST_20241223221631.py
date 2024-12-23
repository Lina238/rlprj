import traci
import numpy as np
import tensorflow as tf
import pickle

class TrafficLightRLTest:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)  # Charger le modèle
        self.STATE_DIM = 6  # Dimension de l'état
        self.ACTION_DIM = 3  # Nombre d'actions possibles (3 phases de feux)
        self.test_data = []  # Liste pour stocker les données de test

    def get_state(self):
        """Obtenir l'état actuel du trafic depuis SUMO"""
        try:
            state = [
                traci.edge.getLastStepHaltingNumber("in"),  # Nombre de véhicules arrêtés sur l'edge "in"
                traci.edge.getLastStepHaltingNumber("E0"),  # Nombre de véhicules arrêtés sur l'edge "E0"
                traci.edge.getLastStepMeanSpeed("in"),  # Vitesse moyenne sur l'edge "in"
                traci.edge.getLastStepMeanSpeed("E0"),  # Vitesse moyenne sur l'edge "E0"
                traci.edge.getLastStepOccupancy("2to3"),  # Occupation de l'edge "2to3"
                float(traci.trafficlight.getPhase("n6"))  # Phase actuelle du feu de signalisation
            ]
            return np.array(state, dtype=np.float32)
        except traci.TraCIException as e:
            print(f"Erreur lors de la récupération de l'état: {e}")
            return np.zeros(self.STATE_DIM, dtype=np.float32)  # Retourner un état vide en cas d'erreur

    def choose_action(self, state):
        """Choisir une action avec le modèle chargé"""
        state_tensor = np.expand_dims(state, 0)  # Ajouter une dimension pour le batch
        q_values = self.model.predict(state_tensor, verbose=0)  # Prédire les Q-valeurs pour l'état donné
        return np.argmax(q_values[0])  # Choisir l'action avec la plus grande Q-valeur

    def test(self, sumo_config):
        """Tester l'agent sur une simulation SUMO"""
        try:
            traci.start(["sumo", "-c", sumo_config])  # Démarrer la simulation SUMO

            state = self.get_state()  # Obtenir l'état initial
            total_reward = 0
            step = 0

            while traci.simulation.getTime() <= 3600:  # Exécuter pendant un certain temps
                action = self.choose_action(state)  # Choisir une action à partir du modèle
                traci.trafficlight.setPhase("n6", action)  # Appliquer la phase choisie

                # Effectuer un pas de simulation
                traci.simulationStep()
                next_state = self.get_state()  # Obtenir l'état suivant
                reward = self.calculate_reward(state, action, next_state)  # Calculer la récompense

                # Sauvegarder les données de test
                self.test_data.append({
                    "state": state.tolist(),
                    "action": action,
                    "reward": reward,
                    "next_state": next_state.tolist()
                })

                state = next_state
                total_reward += reward
                step += 1

            print(f"Test terminé, Récompense totale: {total_reward:.2f}, Étapes: {step}")
        except Exception as e:
            print(f"Erreur pendant le test: {e}")
        finally:
            traci.close()  
            with open("DQL_TEST_VISUALISATION.pkl", "wb") as f:
                pickle.dump(self.test_data, f)

    def calculate_reward(self, state, action, next_state):
        """Calculer la récompense associée à une transition d'état"""
        queue_length = -(next_state[0] + next_state[1])  # Récompense basée sur la longueur des files d'attente
        speed_factor = (next_state[2] + next_state[3]) * 0.5  # Facteur basé sur la vitesse moyenne
        congestion_penalty = -next_state[4] * 3 if next_state[4] > 0.7 else 0  # Pénalité si l'occupation est trop élevée
        phase_change = -5 if state[5] != next_state[5] else 0  # Pénalité si la phase du feu a changé

        return float(queue_length + speed_factor + congestion_penalty + phase_change)

# Tester avec le modèle pré-existant
if __name__ == "__main__":
    sumo_config_file = "../projectnet.sumocfg" 
    model_path = "./DQL_models/traffic_light_control_episode_100.h5" 
    tester = TrafficLightRLTest(model_path)
    tester.test(sumo_config_file)

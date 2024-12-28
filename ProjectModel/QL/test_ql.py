import pickle
import time
import numpy as np
import traci
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

# Charger le modèle
model = tf.keras.models.load_model(
    "trained_traffic_light_model_ql.h5",
    custom_objects={"mse": MeanSquaredError()}
)

# Fonction pour obtenir l'état actuel
def get_state():
    """Récupérer l'état actuel de la circulation depuis SUMO."""
    return [
        traci.edge.getLastStepHaltingNumber("in"),
        traci.edge.getLastStepHaltingNumber("E2"),
        traci.edge.getLastStepMeanSpeed("in"),
        traci.edge.getLastStepMeanSpeed("E2"),
    ]

# Fonction pour calculer la récompense
def calculate_reward(state, next_state):
    """Calculer la récompense pour la transition actuelle."""
    queue_penalty = -(next_state[0] + next_state[1])  # Pénalité de file d'attente
    speed_reward = (next_state[2] + next_state[3]) * 0.5  # Récompense de vitesse
    return queue_penalty + speed_reward

# Connecter SUMO
config_file = r"C:\Users\PRO INFORMATIQUE\Desktop\rl\rlprj\projectnet.sumocfg"
try:
    traci.start(["sumo-gui", "-c", config_file])
    print("SUMO démarré avec succès.")
except Exception as e:
    print(f"Erreur lors du démarrage de SUMO: {e}")
    exit()

# Initialiser les données de test
test_data = []

# Limite d'itérations pour éviter une boucle infinie
iteration_limit = 1000
iteration_count = 0

try:
    # Boucle de test
    while traci.simulation.getMinExpectedNumber() > 0 and iteration_count < iteration_limit:
        # Récupérer l'état actuel
        state = get_state()

        # Prédire les valeurs Q et choisir la meilleure action
        q_values = model.predict(np.array(state).reshape(1, -1), verbose=0)[0]
        action = np.argmax(q_values)

        # Appliquer l'action au feu de circulation
        traci.trafficlight.setPhase("n6", action)

        # Avancer la simulation
        traci.simulationStep()

        # Récupérer l'état suivant
        next_state = get_state()

        # Calculer la récompense
        reward = calculate_reward(state, next_state)

        # Enregistrer les données de test
        test_data.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })

        # Compter le nombre d'itérations
        iteration_count += 1
        time.sleep(0.1)  # Ralentir la simulation pour la visualisation

except Exception as e:
    print(f"Erreur pendant la simulation: {e}")

finally:
    # Sauvegarder les données de test dans un fichier
    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
        print("Données de test sauvegardées dans 'test_data.pkl'.")

    # Fermer la connexion SUMO
    traci.close()
    print("Connexion SUMO fermée.")

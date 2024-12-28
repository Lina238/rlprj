import traci
import pickle
import time
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError  # Importer la fonction de perte

# Charger le modèle entraîné avec la fonction de perte 'mse'
# Charger le modèle
model = tf.keras.models.load_model(
    "trained_traffic_light_model_ql.h5",
    custom_objects={"mse": MeanSquaredError()}
)
# Charger les données de test
try:
    with open(r"C:\Users\PRO INFORMATIQUE\Desktop\rl\rlprj\ProjectModel\QL\test_data.pkl", "rb") as f:
 
        test_data = pickle.load(f)
    print("Données de test chargées avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des données de test: {e}")
    exit()  # Quitter si le chargement échoue

# Connecter à SUMO avec l'interface graphique
try:
    traci.start([r"sumo-gui", "-c", r"C:\Users\PRO INFORMATIQUE\Desktop\rl\rlprj\projectnet.sumocfg"])
    print("Connexion à SUMO réussie.")

    # Boucler à travers les données de test et appliquer les actions
    for entry in test_data:
        state = entry["state"]

        # Utiliser le modèle entraîné pour choisir une action
        q_values = model.predict(np.array(state).reshape(1, -1))[0]
        action = np.argmax(q_values)

        # Appliquer l'action choisie au feu de circulation dans SUMO
        traci.trafficlight.setPhase("n6", action)

        # Passer à l'étape suivante de la simulation
        traci.simulationStep()

        # Optionnel : Ajouter un délai pour ralentir la visualisation
        time.sleep(0.1)  # Ajuster le délai selon les besoins

except traci.exceptions.FatalTraCIError as e:
    print(f"Erreur fatale Traci: {e}")

except Exception as e:
    print(f"Une erreur s'est produite pendant l'exécution: {e}")

finally:
    # Fermer la connexion à SUMO dans le bloc 'finally'
    try:
        traci.close()
        print("Connexion SUMO fermée avec succès.")
    except traci.exceptions.FatalTraCIError:
        print("Aucune connexion SUMO active à fermer.")

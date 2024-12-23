import traci
import numpy as np
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import get_custom_objects

@keras.saving.register_keras_serializable()
class CustomLoss(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.reduce_mean(tf.square(y_true - y_pred))
def evaluate_traffic():
    """
    Évalue les performances du réseau en analysant les statistiques du trafic :
    - Nombre de véhicules bloqués
    - Longueur moyenne des files d'attente
    - Vitesse moyenne des véhicules
    """
    try:
        # Obtenir la liste des véhicules actifs
        vehicle_ids = traci.vehicle.getIDList()
        total_vehicles = len(vehicle_ids)
        
        if total_vehicles == 0:
            print("Aucun véhicule actif dans la simulation.")
            return

        # Initialisation des métriques
        total_queue_length = 0
        total_speed = 0
        blocked_vehicles = 0

        for veh_id in vehicle_ids:
            # Longueur des files d'attente (véhicules arrêtés ou très lents)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:  # Seuil pour considérer un véhicule comme bloqué
                blocked_vehicles += 1

            # Cumul des vitesses pour calculer la moyenne
            total_speed += speed

        # Longueur moyenne des files d'attente
        for edge_id in traci.edge.getIDList():
            total_queue_length += traci.edge.getLastStepHaltingNumber(edge_id)

        avg_queue_length = total_queue_length / len(traci.edge.getIDList())
        avg_speed = total_speed / total_vehicles

        print(f"Total véhicules : {total_vehicles}")
        print(f"Véhicules bloqués : {blocked_vehicles}")
        print(f"Longueur moyenne des files d'attente : {avg_queue_length:.2f}")
        print(f"Vitesse moyenne : {avg_speed:.2f} m/s")

        return {
            "total_vehicles": total_vehicles,
            "blocked_vehicles": blocked_vehicles,
            "avg_queue_length": avg_queue_length,
            "avg_speed": avg_speed
        }

    except traci.TraCIException as e:
        print(f"Erreur lors de l'évaluation du trafic : {e}")


if __name__ == "__main__":
    sumo_config = "../projectnet.sumocfg"
    
    try:
        traci.start(["sumo", "-c", sumo_config])
        step = 0

        while step < 1000:  # Simulation limitée à 1000 étapes
            traci.simulationStep()

            if step % 100 == 0:  # Évaluation toutes les 100 étapes
                metrics = evaluate_traffic()
                print(metrics)

            step += 1

    except Exception as e:
        print(f"Erreur dans la simulation : {e}")
    finally:
        traci.close()
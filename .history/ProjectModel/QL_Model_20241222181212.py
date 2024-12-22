import traci
import numpy as np
import tensorflow as tf


# Se connecter à SUMO
traci.start(["sumo", "-c", "../projectnet.sumocfg"])

try:
    # Déclaration des constantes selon notre network (projectnet.net)
    state = [
        traci.edge.getLastStepHaltingNumber("in"),       # Nombre de véhicules en arrêt sur la route principale
        traci.edge.getLastStepHaltingNumber("E2"),  # Nombre de véhicules en arrêt sur la rampe
        traci.edge.getLastStepMeanSpeed("in"),           # Vitesse moyenne des véhicules sur la route principale
        traci.edge.getLastStepMeanSpeed("E2"),      # Vitesse moyenne des véhicules sur la rampe
    ]
    
    # Afficher les états
    print("States:", state)
# Model de Q-Learning

finally:
    # Fermer la connexion
    traci.close()
# on va avoir States: [0, 0, 30.0, 30.0]
'''[0, 0, 30.0, 30.0] :

0: Nombre de véhicules en arrêt sur la route principale (in).
0: Nombre de véhicules en arrêt sur la rampe (intramp).
30.0: Vitesse moyenne des véhicules sur la route principale (in).
30.0: Vitesse moyenne des véhicules sur la rampe (intramp).'''
# pour faire marcher les flows:
# import traci

# # Démarrer la simulation
# traci.start(["sumo", "-c", "../projectnet.sumocfg"])

# # Étape de simulation
# for step in range(300):  # 300 correspond à la durée de votre flow
#     traci.simulationStep()

#     # Obtenir la liste des véhicules présents à cette étape
#     vehicles = traci.vehicle.getIDList()
#     print(f"Étape {step}: {len(vehicles)} véhicules présents.")
    
#     # Afficher les positions des véhicules
#     for vehicle_id in vehicles:
#         position = traci.vehicle.getPosition(vehicle_id)
#         print(f"Véhicule {vehicle_id} à la position {position}")

# # Fermer la connexion
# traci.close()

import traci
import numpy as np
import tensorflow as tf
import pickle
import time
import subprocess

#délaration des constantes selon notre network (projectnet.net)
state = [
    traci.edge.getLastStepHaltingNumber("in"),       # Nombre de véhicules en arrêt sur la route principale
    traci.edge.getLastStepHaltingNumber("intramp"),  # Nombre de véhicules en arrêt sur la rampe
    traci.edge.getLastStepMeanSpeed("in"),           # Vitesse moyenne des véhicules sur la route principale
    traci.edge.getLastStepMeanSpeed("intramp"),      # Vitesse moyenne des véhicules sur la rampe
]

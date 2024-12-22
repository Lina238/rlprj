'''traci : Interface pour interagir avec SUMO, permettant de contrôler
les feux de circulation et d'extraire des données en temps réel.
dql:
c'est une commi
'''
import traci
import tensorflow as tf
import numpy as np
traci.start(["sumo", "-c", "../projectnet.sumocfg"])

'''traci : Interface pour interagir avec SUMO, permettant de contrôler
les feux de circulation et d'extraire des données en temps réel.
dql:
c'est une combinaison entre le double q-learning et le NN ,Il aide à résoudre les problèmes suivants:
-le changement de toutes les valeurs des hidden layers:il utilise 2 NN un qui change ces valeurs
lors de la backpropagation et l'autre (qui est le target network ) qui reste fixe pour 
un certain nombre d'itérations
'''
import traci
import tensorflow as tf
import numpy as np
traci.start(["sumo", "-c", "../projectnet.sumocfg"])

'''traci : Interface pour interagir avec SUMO, permettant de contrôler
les feux de circulation et d'extraire des données en temps réel.
dql:
c'est une combinaison entre le double q-learning et le NN ,Il aide à résoudre les problèmes suivants:
-le changement de toutes les valeurs des hidden layers ce qui engendre des oscillations :il utilise 2 NN un(Une copie du réseau principal est utilisée pour calculer les Q' valeurs
) qui change ces valeurs
lors de la backpropagation et l'autre (qui est le target network ) qui change que les valeurs qui nous aurions 
réelement besoin de changer
-les valeurs de Q qui sont corrélées entre elles:il utilise un buffer de replay qui stocke les
valeurs de Q
'''


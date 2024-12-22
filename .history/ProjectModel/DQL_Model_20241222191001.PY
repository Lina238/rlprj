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
import traci
import numpy as np
import tensorflow as tf
import os

class TrafficLightRL:
    """
    Classe principale pour le contrôle intelligent des feux de signalisation
    utilisant l'apprentissage par renforcement profond (Deep Q-Learning)
    """
    def __init__(self):
        # Étape 1: Initialisation des paramètres constants
        # ------------------------------------------
        # Paramètres du système de feux
        self.NUM_PHASES = 3        # Nombre de phases possibles (vert, jaune, rouge)
        self.STATE_DIM = 6         # Dimension de l'état (nombre de mesures utilisées)
        self.ACTION_DIM = self.NUM_PHASES  # Nombre d'actions possibles
        
        # Paramètres d'apprentissage
        self.GAMMA = 0.9          # Facteur de dépréciation pour les récompenses futures
        self.EPSILON = 0.1        # Probabilité d'exploration (vs exploitation)
        self.ALPHA = 0.001        # Taux d'apprentissage du réseau
        self.MEMORY_CAPACITY = 10000  # Taille de la mémoire d'expérience
        self.BATCH_SIZE = 32      # Taille des lots pour l'apprentissage
        self.NUM_EPISODES = 100   # Nombre total d'épisodes d'entraînement
        self.TARGET_UPDATE_FREQ = 10  # Fréquence de mise à jour du réseau cible
        self.SIMULATION_TIME = 3600   # Durée de simulation en secondes (1 heure)

        # Étape 2: Initialisation des réseaux de neurones
        # ------------------------------------------
        self.model = self._build_model()        # Réseau principal
        self.target_model = self._build_model() # Réseau cible
        self.target_model.set_weights(self.model.get_weights())
        self.memory = []  # Mémoire d'expérience

    def _build_model(self):
        """
        Construction du réseau de neurones pour l'apprentissage
        Architecture: Entrée -> 64 neurones -> 64 neurones -> Sortie
        """
        model = tf.keras.Sequential([
            # Couche d'entrée avec activation ReLU
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.STATE_DIM,)),
            # Couche cachée
            tf.keras.layers.Dense(64, activation='relu'),
            # Couche de sortie (une sortie par action possible)
            tf.keras.layers.Dense(self.ACTION_DIM, activation='linear')
        ])
        # Configuration de l'optimiseur et de la fonction de perte
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.ALPHA), loss='mse')
        return model

    def get_state(self):
        """
        Récupération de l'état actuel du trafic
        Retourne un vecteur contenant 6 mesures différentes
        """
        return np.array([
            self._get_edge_data("in"),      # Nombre de véhicules arrêtés sur la route principale
            self._get_edge_data("E0"),      # Nombre de véhicules arrêtés sur la rampe
            self._get_speed_data("in"),     # Vitesse moyenne sur la route principale
            self._get_speed_data("E0"),     # Vitesse moyenne sur la rampe
            self._get_occupancy_data("2to3"),  # Taux d'occupation après la fusion
            traci.trafficlight.getPhase("n6")  # Phase actuelle du feu
        ])

    def _get_edge_data(self, edge_id):
        """Récupère le nombre de véhicules arrêtés sur un tronçon"""
        return traci.edge.getLastStepHaltingNumber(edge_id)

    def _get_speed_data(self, edge_id):
        """Récupère la vitesse moyenne sur un tronçon"""
        return traci.edge.getLastStepMeanSpeed(edge_id)

    def _get_occupancy_data(self, edge_id):
        """Récupère le taux d'occupation d'un tronçon"""
        return traci.edge.getLastStepOccupancy(edge_id)

    def choose_action(self, state):
        """
        Sélection d'une action selon la politique epsilon-greedy
        Args:
            state: État actuel du système
        Returns:
            action: Indice de la phase choisie
        """
        # Exploration: action aléatoire avec probabilité epsilon
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.ACTION_DIM)
        # Exploitation: meilleure action selon le réseau
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)

    def calculate_reward(self, state, action, next_state):
        """
        Calcul de la récompense basée sur plusieurs métriques de trafic
        
        Composantes de la récompense:
        1. Pénalité pour les files d'attente
        2. Bonus pour la fluidité du trafic
        3. Pénalité pour la congestion
        4. Pénalité pour les changements de phase fréquents
        """
        # Calcul des différentes composantes
        queue_length = -(next_state[0] + next_state[1])  # Pénalité pour les files d'attente
        speed_factor = (next_state[2] + next_state[3]) * 0.5  # Récompense pour la vitesse
        congestion_penalty = -next_state[4] * 3 if next_state[4] > 0.7 else 0  # Pénalité de congestion
        phase_change = -5 if state[5] != next_state[5] else 0  # Pénalité de changement de phase
        
        return queue_length + speed_factor + congestion_penalty + phase_change

    def update_network(self):
        """
        Mise à jour du réseau de neurones à partir des expériences stockées
        Utilise la technique du Double DQN pour plus de stabilité
        """
        # Vérification de la taille minimum du batch
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sélection aléatoire d'un batch d'expériences
        batch = np.array(self.memory)[np.random.choice(len(self.memory), self.BATCH_SIZE, replace=False)]
        
        # Décomposition du batch
        states = np.vstack(batch[:, 0])
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2]
        next_states = np.vstack(batch[:, 3])
        terminals = batch[:, 4]

        # Calcul des valeurs Q actuelles et futures
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Mise à jour des valeurs Q selon l'équation de Bellman
        for i in range(self.BATCH_SIZE):
            if terminals[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.GAMMA * np.max(next_q_values[i])

        # Entraînement du réseau
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def train(self, sumo_config):
        """
        Boucle principale d'entraînement
        
        Étapes pour chaque épisode:
        1. Initialisation de la simulation
        2. Collecte d'expériences
        3. Apprentissage
        4. Mise à jour du réseau cible
        5. Sauvegarde périodique du modèle
        """
        try:
            # Démarrage de SUMO
            traci.start(["sumo", "-c", sumo_config])
            
            for episode in range(self.NUM_EPISODES):
                # Réinitialisation de la simulation
                traci.load(["-c", sumo_config])
                state = self.get_state()
                total_reward = 0
                step = 0

                # Boucle de simulation
                while traci.simulation.getTime() <= self.SIMULATION_TIME:
                    # 1. Sélection et application de l'action
                    action = self.choose_action(state)
                    self._apply_action(action)
                    
                    # 2. Observation du nouvel état et calcul de la récompense
                    next_state = self.get_state()
                    reward = self.calculate_reward(state, action, next_state)
                    done = traci.simulation.getTime() > self.SIMULATION_TIME

                    # 3. Stockage de l'expérience et apprentissage
                    self._store_experience(state, action, reward, next_state, done)
                    self.update_network()

                    # 4. Préparation pour la prochaine étape
                    state = next_state
                    total_reward += reward
                    step += 1

                    if done:
                        break

                # Affichage des résultats de l'épisode
                print(f"Épisode {episode + 1}/{self.NUM_EPISODES}, Récompense totale: {total_reward:.2f}")
                
                # Mise à jour du réseau cible
                if episode % self.TARGET_UPDATE_FREQ == 0:
                    self.target_model.set_weights(self.model.get_weights())
                
                # Sauvegarde périodique du modèle
                if (episode + 1) % 10 == 0:
                    self._save_model(episode)

        finally:
            # Fermeture propre de SUMO
            traci.close()

    def _apply_action(self, action):
        """
        Application de l'action choisie au feu de circulation
        Gestion des durées de phase selon le type de phase
        """
        traci.trafficlight.setPhase("n6", action)
        phase_duration = self._get_phase_duration(action)
        for _ in range(phase_duration):
            traci.simulationStep()

    def _get_phase_duration(self, action):
        """
        Définition des durées pour chaque phase
        0: Vert (82s), 1: Jaune (3s), 2: Rouge (5s)
        """
        durations = {0: 82, 1: 3, 2: 5}
        return durations.get(action, 5)

    def _store_experience(self, state, action, reward, next_state, done):
        """
        Stockage d'une nouvelle expérience dans la mémoire
        Suppression de la plus ancienne si la capacité est atteinte
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.MEMORY_CAPACITY:
            self.memory.pop(0)

    def _save_model(self, episode):
        """
        Sauvegarde du modèle dans un fichier
        Création du dossier de sauvegarde si nécessaire
        """
        save_dir = "models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(os.path.join(save_dir, f"traffic_model_episode_{episode + 1}.h5"))

# Point d'entrée du programme
def main():
    """
    Fonction principale d'exécution
    Configuration et lancement de l'apprentissage
    """
    config_file = "../projectnet.sumocfg"
    agent = TrafficLightRL()
    agent.train(config_file)

if __name__ == "__main__":
    main()
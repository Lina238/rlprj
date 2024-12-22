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
from collections import deque
import random

class TrafficLightRL:
    def __init__(self):
        # Paramètres constants
        self.NUM_PHASES = 3
        self.STATE_DIM = 6
        self.ACTION_DIM = self.NUM_PHASES
        self.GAMMA = 0.9
        self.EPSILON = 0.1
        # self.EPSILON = max(0.01, self.EPSILON * 0.995)  # Décroît avec chaque épisode

        self.ALPHA = 0.001
        self.MEMORY_CAPACITY = 10000
        self.BATCH_SIZE = 32
        self.NUM_EPISODES = 100
        self.TARGET_UPDATE_FREQ = 10
        self.SIMULATION_TIME = 3600

        # Initialisation des réseaux
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Utilisation de deque pour la mémoire
        self.memory = deque(maxlen=self.MEMORY_CAPACITY)

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.STATE_DIM,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.ACTION_DIM, activation='linear')
        ])
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.ALPHA), loss='mse')
        return model

    def get_state(self):
        try:
            state = [
                traci.edge.getLastStepHaltingNumber("in"),
                traci.edge.getLastStepHaltingNumber("E0"),
                traci.edge.getLastStepMeanSpeed("in"),
                traci.edge.getLastStepMeanSpeed("E0"),
                traci.edge.getLastStepOccupancy("2to3"),
                float(traci.trafficlight.getPhase("n6"))
            ]
            return np.array(state, dtype=np.float32)
        except traci.TraCIException as e:
            print(f"Erreur lors de la récupération de l'état: {e}")
            return np.zeros(self.STATE_DIM, dtype=np.float32)

    def choose_action(self, state):
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.ACTION_DIM)
        state_tensor = np.expand_dims(state, 0)
        q_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def calculate_reward(self, state, action, next_state):
        queue_length = -(next_state[0] + next_state[1])
        speed_factor = (next_state[2] + next_state[3]) * 0.5
        congestion_penalty = -next_state[4] * 3 if next_state[4] > 0.7 else 0
        phase_change = -5 if state[5] != next_state[5] else 0
        
        return float(queue_length + speed_factor + congestion_penalty + phase_change)

    def _store_experience(self, state, action, reward, next_state, done):
        """Stockage sécurisé des expériences"""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def update_network(self):
        """Mise à jour du réseau avec gestion correcte des batchs"""
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Échantillonnage aléatoire du batch
        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        
        # Préparation des arrays
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Calcul des valeurs Q actuelles et futures
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Mise à jour des valeurs Q
        targets = current_q_values.copy()
        for i in range(self.BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.GAMMA * np.max(next_q_values[i])

        # Entraînement du modèle
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.BATCH_SIZE)

    def train(self, sumo_config,control_traffic_lights=True,model_name="traffic_light_model"):
        """Boucle d'entraînement principale avec gestion des erreurs"""
        try:
            traci.start(["sumo", "-c", sumo_config])
            
            for episode in range(self.NUM_EPISODES):
                print(f"Début de l'épisode {episode + 1}/{self.NUM_EPISODES}")
                traci.load(["-c", sumo_config])
                state = self.get_state()
                total_reward = 0
                step = 0

                while traci.simulation.getTime() <= self.SIMULATION_TIME:
                 if control_traffic_lights:
                    # Sélection et application de l'action
                    action = self.choose_action(state)
                    self._apply_action(action)
                 else:
                    # Maintient les feux toujours verts
                    traci.trafficlight.setPhase("n6", 0)
                    traci.simulationStep()   
                    # Observation et récompense
                next_state = self.get_state()
                reward = self.calculate_reward(state, action, next_state)
                done = traci.simulation.getTime() > self.SIMULATION_TIME

                # Stockage et apprentissage
                self._store_experience(state, action, reward, next_state, done)
                self.update_network()

                state = next_state
                total_reward += reward
                step += 1

                if done:
                        break

                print(f"Épisode {episode + 1} terminé, Récompense: {total_reward:.2f}, Étapes: {step}")
                
                if episode % self.TARGET_UPDATE_FREQ == 0:
                    self.target_model.set_weights(self.model.get_weights())
                
                if (episode + 1) % 10 == 0:
                    self._save_model(model_name, episode)

        except Exception as e:
            print(f"Erreur pendant l'entraînement: {e}")
        finally:
            traci.close()

    def _apply_action(self, action):
        try:
            traci.trafficlight.setPhase("n6", action)
            phase_duration = self._get_phase_duration(action)
            for _ in range(phase_duration):
                traci.simulationStep()
        except traci.TraCIException as e:
            print(f"Erreur lors de l'application de l'action: {e}")

    def _get_phase_duration(self, action):
        durations = {0: 82, 1: 3, 2: 5}
        return durations.get(action, 5)

    def _save_model(self, model_name, episode):
       """Sauvegarde du modèle sous un nom spécifique."""
       save_dir = "models"
       if not os.path.exists(save_dir):
        os.makedirs(save_dir)
       self.model.save(os.path.join(save_dir, f"{model_name}_episode_{episode + 1}.h5"))
def main():
    config_file = "../projectnet.sumocfg"
    agent = TrafficLightRL()
    agent.train(config_file,)

if __name__ == "__main__":
    main()
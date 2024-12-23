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
import json
class TrafficLightRL:
    def __init__(self):
        # Paramètres constants
        self.NUM_PHASES = 3  # Nombre de phases possibles pour les feux de signalisation
        self.STATE_DIM = 6  # Dimension de l'état (nombre de variables utilisées)
        self.ACTION_DIM = self.NUM_PHASES  # Nombre d'actions possibles (phases de feux)
        self.GAMMA = 0.9  # Facteur d'actualisation pour les récompenses futures
        self.EPSILON = 1.0  # Valeur initiale pour EPSILON (exploration)
        self.EPSILON_DECAY = 0.995  # Facteur de décroissance de l'exploration
        self.MIN_EPSILON = 0.01  # Valeur minimale de EPSILON
        self.rewards_history = []
        self.ALPHA = 0.001  # Taux d'apprentissage pour l'optimisation du modèle
        self.MEMORY_CAPACITY = 10000  # Taille maximale de la mémoire pour les expériences
        self.BATCH_SIZE = 32  # Taille des mini-lots pour l'entraînement
        self.NUM_EPISODES = 100  # Nombre d'épisodes pour l'entraînement
        self.TARGET_UPDATE_FREQ = 10  # Fréquence de mise à jour du modèle cible
        self.SIMULATION_TIME = 3600  # Durée de la simulation en secondes

        # Initialisation des réseaux neuronaux
        self.model = self._build_model()  # Modèle principal
        self.target_model = self._build_model()  # Modèle cible
        self.target_model.set_weights(self.model.get_weights())  # Initialisation des poids du modèle cible

        # Utilisation de deque pour la mémoire (circular buffer)
        self.memory = deque(maxlen=self.MEMORY_CAPACITY)
    class EpisodeInfo:
        def __init__(self, episode, reward, steps,loss):
            self.episode = episode
            self.reward = reward
            self.steps = steps
            self.loss = loss 
    def _build_model(self):
        """Construire le modèle de réseau neuronal"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.STATE_DIM,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.ACTION_DIM, activation='linear')
        ])
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.ALPHA), loss='mse')
        return model

    def get_state(self):
        """Obtenir l'état actuel du trafic depuis SUMO"""
        try:
            state = [
                traci.edge.getLastStepHaltingNumber("in"),  # Nombre de véhicules arrêtés sur l'edge "in"
                traci.edge.getLastStepHaltingNumber("E0"),  # Nombre de véhicules arrêtés sur l'edge "E0"
                traci.edge.getLastStepMeanSpeed("in"),  # Vitesse moyenne sur l'edge "in"
                traci.edge.getLastStepMeanSpeed("E0"),  # Vitesse moyenne sur l'edge "E0"
                traci.edge.getLastStepOccupancy("2to3"),  # Occupation de l'edge "2to3"
                float(traci.trafficlight.getPhase("n6"))  # Phase actuelle du feu de signalisation
            ]
            return np.array(state, dtype=np.float32)
        except traci.TraCIException as e:
            print(f"Erreur lors de la récupération de l'état: {e}")
            return np.zeros(self.STATE_DIM, dtype=np.float32)  # Retourner un état vide en cas d'erreur

    def choose_action(self, state):
        """Choisir une action selon la politique epsilon-greedy"""
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.ACTION_DIM)  # Exploration : choisir une action aléatoire
        state_tensor = np.expand_dims(state, 0)  # Ajouter une dimension pour le batch
        q_values = self.model.predict(state_tensor, verbose=0)  # Prédire les Q-valeurs pour l'état donné
        return np.argmax(q_values[0])  # Choisir l'action avec la plus grande Q-valeur
    def save_rewards(self, filename="rewards_history.json"):
        """Sauvegarder les récompenses de tous les épisodes dans un fichier JSON."""
        try:
            # Convertir les objets EpisodeInfo en dictionnaire pour pouvoir les sérialiser en JSON
            rewards_data = [
                {"episode": ep.episode, "reward": ep.reward, "steps": ep.steps,"loss": ep.loss}
                for ep in self.rewards_history
            ]
            with open(filename, 'w') as f:
                json.dump(rewards_data, f, indent=4)
            print(f"Récompenses sauvegardées dans {filename}")
        
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des récompenses : {e}")
    def calculate_reward(self, state, action, next_state):
        """Calculer la récompense associée à une transition d'état"""
        queue_length = -(next_state[0] + next_state[1])  # Récompense basée sur la longueur des files d'attente
        speed_factor = (next_state[2] + next_state[3]) * 0.5  # Facteur basé sur la vitesse moyenne
        congestion_penalty = -next_state[4] * 3 if next_state[4] > 0.7 else 0  # Pénalité si l'occupation est trop élevée
        phase_change = -5 if state[5] != next_state[5] else 0  # Pénalité si la phase du feu a changé
        
        return float(queue_length + speed_factor + congestion_penalty + phase_change)

    def _store_experience(self, state, action, reward, next_state, done):
        """Enregistrer l'expérience dans la mémoire"""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))  # Ajouter l'expérience dans la mémoire

    def update_network(self):
        """Mettre à jour les poids du réseau de neurones"""
        if len(self.memory) < self.BATCH_SIZE:  # S'il n'y a pas assez d'expériences
            return

        minibatch = random.sample(self.memory, self.BATCH_SIZE)  # Prendre un mini-lot d'expériences

        # Extraire les composants du mini-lot
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        current_q_values = self.model.predict(states, verbose=0)  # Prédire les Q-valeurs actuelles
        next_q_values = self.target_model.predict(next_states, verbose=0)  # Prédire les Q-valeurs futures

        targets = current_q_values.copy()  # Copier les Q-valeurs actuelles
        for i in range(self.BATCH_SIZE):
            if dones[i]:  # Si l'épisode est terminé
                targets[i][actions[i]] = rewards[i]
            else:  # Sinon, mise à jour avec la récompense et les Q-valeurs futures
                targets[i][actions[i]] = rewards[i] + self.GAMMA * np.max(next_q_values[i])

        # Entraînement du modèle avec les cibles mises à jour
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.BATCH_SIZE)
        loss = history.history['loss'][0]  # La première (et unique) valeur de la perte
        return loss
    def train(self, sumo_config, control_traffic_lights=True, model_name="traffic_light_model"):
        """Entraîner l'agent en simulant les épisodes dans SUMO"""
        try:
         traci.start(["sumo", "-c", sumo_config])  # Démarrer la simulation SUMO
        
         for episode in range(self.NUM_EPISODES):
            print(f"Début de l'épisode {episode + 1}/{self.NUM_EPISODES}")
            traci.load(["-c", sumo_config])  # Recharger la simulation pour chaque épisode
            state = self.get_state()  # Obtenir l'état initial
            total_reward = 0
            total_loss = 0  # Variable pour suivre la perte totale
            step = 0

            while traci.simulation.getTime() <= self.SIMULATION_TIME:  # Exécuter l'épisode pendant un certain temps
                if control_traffic_lights:  # Si le contrôle des feux est activé
                    action = self.choose_action(state)  # Choisir l'action à partir de la politique
                    self._apply_action(action)  # Appliquer l'action (changer la phase du feu)
                else:
                    traci.trafficlight.setPhase("n6", 0)  # Phase par défaut
                    traci.simulationStep()  # Effectuer un pas de simulation
                
                next_state = self.get_state()  # Obtenir l'état suivant
                reward = self.calculate_reward(state, action, next_state)  # Calculer la récompense
                done = traci.simulation.getTime() > self.SIMULATION_TIME  # Vérifier si l'épisode est terminé

                self._store_experience(state, action, reward, next_state, done)  # Enregistrer l'expérience
                loss = self.update_network()  # Mettre à jour le modèle et obtenir la perte

                state = next_state
                total_reward += reward
                total_loss += loss  # Accumuler la perte
                step += 1

                if done:
                    break

            print(f"Épisode {episode + 1} terminé, Récompense: {total_reward:.2f}, Perte: {total_loss:.4f}, Étapes: {step}")
            episode_info = self.EpisodeInfo(episode + 1, total_reward, step, total_loss)
            self.rewards_history.append(episode_info)
            self.save_rewards()
            if episode % self.TARGET_UPDATE_FREQ == 0:  # Mise à jour du modèle cible
                self.target_model.set_weights(self.model.get_weights())
            
            if (episode + 1) % 10 == 0:  # Sauvegarder le modèle tous les 10 épisodes
                self._save_model(model_name, episode)

    except Exception as e:
        print(f"Erreur pendant l'entraînement: {e}")
    finally:
        traci.close()  # Fermer la connexion à SUMO

    def _apply_action(self, action):
        """Appliquer l'action en modifiant la phase du feu"""
        try:
            traci.trafficlight.setPhase("n6", action)  # Appliquer la phase de feu choisie
            phase_duration = self._get_phase_duration(action)  # Durée de la phase de feu
            for _ in range(phase_duration):
                traci.simulationStep()  # Avancer dans la simulation
        except traci.TraCIException as e:
            print(f"Erreur lors de l'application de l'action: {e}")

    def _get_phase_duration(self, action):
        """Retourner la durée de la phase en fonction de l'action"""
        durations = {0: 82, 1: 3, 2: 5}  # Durées de chaque phase de feu
        return durations.get(action, 5)  # Retourner la durée associée à l'action

    def _save_model(self, model_name, episode):
        """Sauvegarder le modèle à chaque épisode"""
        save_dir = "DQL_models"
        if not os.path.exists(save_dir):  # Créer le répertoire si nécessaire
            os.makedirs(save_dir)
        self.model.save(os.path.join(save_dir, f"{model_name}_episode_{episode + 1}.h5"))  # Sauvegarder le modèle

def main():
    """Fonction principale d'entraînement"""
    config_file = "../projectnet.sumocfg"  # Fichier de configuration SUMO
    agent = TrafficLightRL()  # Créer une instance de l'agent
    agent.train(config_file, control_traffic_lights=True, model_name="traffic_light_control")  # Entraîner l'agent

if __name__ == "__main__":
    main()  # Exécuter la fonction principale

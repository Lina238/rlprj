import traci
import numpy as np
import tensorflow as tf
import pickle
import time
import subprocess



import numpy as np
import tensorflow as tf
import random
from collections import deque
import traci

# Constants
NUM_PHASES = 4  # Number of traffic light phases
STATE_DIM = 4  # Number of state variables (adjust based on your scenario)
ACTION_DIM = NUM_PHASES  # Number of possible actions (traffic light phases)
GAMMA = 0.9  # Discount factor
# EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.1  # Minimum exploration rate
# EPSILON_DECAY = 0.995  # Decay rate for exploration
ALPHA = 0.0001  # Learning rate
MEMORY_CAPACITY = 10000  # Replay memory capacity
BATCH_SIZE = 32  # Batch size for training
NUM_EPISODES = 100  # Number of training episodes

# Q-network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(STATE_DIM,)),
    tf.keras.layers.Dense(ACTION_DIM, activation="linear"),
])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=ALPHA), loss="mse")

# Target Q-network (used for more stable updates)
target_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(STATE_DIM,)),
    tf.keras.layers.Dense(ACTION_DIM, activation="linear"),
])
target_model.set_weights(model.get_weights())  # Initialize with the same weights
target_model.compile(optimizer=tf.optimizers.Adam(learning_rate=ALPHA), loss="mse")

# Replay memory
memory = deque(maxlen=MEMORY_CAPACITY)

# Epsilon for exploration
epsilon = EPSILON_MIN

def get_state():
    """Implement logic to get the current state from SUMO"""
    state = [
        traci.edge.getLastStepHaltingNumber("in"),  # Main road
        traci.edge.getLastStepHaltingNumber("intramp"),  # Ramp
        traci.edge.getLastStepMeanSpeed("in"),  # Main road speed
        traci.edge.getLastStepMeanSpeed("intramp"),  # Ramp speed
    ]
    return np.array(state)

def choose_action(state):
    """Choose an action using epsilon-greedy policy with epsilon decay"""
    if np.random.rand() < epsilon:
        return np.random.randint(ACTION_DIM)  # Exploration: random action
    else:
        q_values = model.predict(state.reshape(1, -1))[0]
        return np.argmax(q_values)  # Exploitation: action with max Q-value

def update_q_network():
    """Update the Q-network by sampling a minibatch from memory and training the model"""
    if len(memory) < BATCH_SIZE:
        return

    minibatch = random.sample(memory, BATCH_SIZE)

    # Extract each part of the minibatch: state, action, reward, next_state, done
    states = np.array([entry[0] for entry in minibatch])
    actions = np.array([entry[1] for entry in minibatch])
    rewards = np.array([entry[2] for entry in minibatch])
    next_states = np.array([entry[3] for entry in minibatch])
    dones = np.array([entry[4] for entry in minibatch])

    # Predict Q-values for states and next states
    q_values = model.predict(states)
    next_q_values = target_model.predict(next_states)

    # Update Q-values based on the Bellman equation
    for i in range(BATCH_SIZE):
        if dones[i]:
            q_values[i, actions[i]] = rewards[i]  # Terminal state
        else:
            q_values[i, actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])  # Non-terminal state

    # Train the model
    model.fit(states, q_values, epochs=1, verbose=0)

def calculate_reward(state, action, next_state):
    """Custom reward function with more complexity"""
    main_road_halting = state[0]
    ramp_halting = state[1]
    main_road_speed = state[2]
    ramp_speed = state[3]

    # Example of complex reward function:
    reward = -(main_road_halting + ramp_halting) + (main_road_speed + ramp_speed)
    
    # Additional reward conditions (e.g., for green lights, speed optimization)
    if action == 0:  # Assume 0 represents a green phase
        reward += 10  # Reward for keeping traffic moving
    
    return reward

def check_if_done():
    """Termination condition: End the episode after a certain number of simulation steps"""
    return traci.simulation.getTime() > 100  # Example: end after 100 simulation steps

# def update_epsilon():
#     """Decay epsilon after each episode"""
#     global epsilon
#     if epsilon > EPSILON_MIN:
#         epsilon *= EPSILON_DECAY

# Initialize connection to SUMO
traci.start([r"sumo-gui", "-c", r"C:\Users\PRO INFORMATIQUE\Desktop\rl\projet\rlprj\projectnet.sumocfg"])

episode_rewards = []  # To store the total rewards of each episode

# Training loop
for episode in range(NUM_EPISODES):
    state = get_state()
    total_reward = 0

    while traci.simulation.getMinExpectedNumber() > 0:  # Continue until simulation ends
        action = choose_action(state)  # Choose action using epsilon-greedy policy

        # Apply the chosen action to the traffic light in SUMO
        traci.trafficlight.setPhase("n2", action)

        # Step the simulation (allowing the change to take effect)
        traci.simulationStep()

        # Obtain the next state from SUMO
        next_state = get_state()

        # Calculate the reward
        reward = calculate_reward(state, action, next_state)

        # Check if the simulation is done
        done = check_if_done()

        # Log relevant information
        print(f"Step: {traci.simulation.getTime()}, Action: {action}, Reward: {reward}")

        # Store the transition in memory
        memory.append((state, action, reward, next_state, done))

        # Update the Q-network
        update_q_network()

        # Update the current state
        state = next_state
        total_reward += reward

        # If done, break the loop
        if done:
            break

    # Periodically update the target model
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    # Save the model periodically
    if (episode + 1) % 10 == 0:
        model.save("trained_traffic_light_model_ql.h5")

    episode_rewards.append(total_reward)  # Append the total reward for the episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Update epsilon for exploration decay
    # update_epsilon()

# Close connection to SUMO after all episodes
traci.close()

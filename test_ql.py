import traci
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics

# Enregistrez explicitement 'mse' si nécessaire
def custom_mse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

# Chargement du modèle en précisant la métrique 'mse'
model = tf.keras.models.load_model("trained_traffic_light_model_ql.h5", 
                                   custom_objects={"mse": custom_mse})


# Connect to SUMO
try:
    traci.start([r"sumo-gui", "-c", r"C:\Users\PRO INFORMATIQUE\Desktop\rl\projet\rlprj\projectnet.sumocfg"])


    # Initialize the test data list
    test_data = []

    # Testing loop
    while traci.simulation.getMinExpectedNumber() > 0:
        state = [
            traci.edge.getLastStepHaltingNumber("in"),
            traci.edge.getLastStepHaltingNumber("intramp"),
            traci.edge.getLastStepMeanSpeed("in"),
            traci.edge.getLastStepMeanSpeed("intramp"),
        ]

        # Choose action using the trained model
        q_values = model.predict(np.array(state).reshape(1, -1))[0]
        action = np.argmax(q_values)

        # Apply the chosen action to the traffic light in SUMO
        traci.trafficlight.setPhase("n2", action)

        # Step the simulation
        traci.simulationStep()

        # Get the next state
        next_state = [
            traci.edge.getLastStepHaltingNumber("in"),
            traci.edge.getLastStepHaltingNumber("intramp"),
            traci.edge.getLastStepMeanSpeed("in"),
            traci.edge.getLastStepMeanSpeed("intramp"),
        ]

        # Calculate the reward (if needed)
        reward = 0  # You can modify this according to your reward function

        # Log relevant information
        test_data.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })

        # Optional: Add a delay to slow down the visualization
        time.sleep(0.1)  # Adjust the delay as needed

except traci.exceptions.FatalTraCIError as e:
    print(f"Error: {e}")

finally:
    # Close connection to SUMO in the finally block
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass  # Ignore if there is no active connection

# Save the test data to a file
with open("test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)

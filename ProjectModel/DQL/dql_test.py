import traci
import numpy as np
import tensorflow as tf
import pickle

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def get_config(self):
        return {"name": self.name}

class TrafficLightRLTest:
    def __init__(self, model_path):
        # Register both the custom loss and the MSE metric
        custom_objects = {
            "CustomLoss": CustomLoss,
            "mse": tf.keras.metrics.mean_squared_error,
        }
        
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        self.STATE_DIM = 6
        self.ACTION_DIM = 3
        self.test_data = []

    def get_state(self):
        """Get current traffic state from SUMO"""
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
            print(f"Error getting state: {e}")
            return np.zeros(self.STATE_DIM, dtype=np.float32)

    def choose_action(self, state):
        """Choose action using loaded model"""
        state_tensor = np.expand_dims(state, 0)
        try:
            q_values = self.model.predict(state_tensor, verbose=0)
            return np.argmax(q_values[0])
        except Exception as e:
            print(f"Error predicting action: {e}")
            return 0

    def test(self, sumo_config):
        """Test agent on SUMO simulation"""
        try:
            traci.start(["sumo", "-c", sumo_config])
            state = self.get_state()
            total_reward = 0
            step = 0

            while traci.simulation.getTime() <= 3600:
                action = self.choose_action(state)
                traci.trafficlight.setPhase("n6", action)

                traci.simulationStep()
                next_state = self.get_state()
                reward = self.calculate_reward(state, action, next_state)

                self.test_data.append({
                    "state": state.tolist(),
                    "action": action,
                    "reward": reward,
                    "next_state": next_state.tolist()
                })

                state = next_state
                total_reward += reward
                step += 1

            print(f"Test completed, Total reward: {total_reward:.2f}, Steps: {step}")
        except Exception as e:
            print(f"Error during test: {e}")
        finally:
            traci.close()
            try:
                with open("DQL_TEST_VISUALISATION.pkl", "wb") as f:
                    pickle.dump(self.test_data, f)
            except Exception as e:
                print(f"Error saving test data: {e}")

    def calculate_reward(self, state, action, next_state):
        """Calculate reward for state transition"""
        queue_length = -(next_state[0] + next_state[1])
        speed_factor = (next_state[2] + next_state[3]) * 0.5
        congestion_penalty = -next_state[4] * 3 if next_state[4] > 0.7 else 0
        phase_change = -5 if state[5] != next_state[5] else 0

        return float(queue_length + speed_factor + congestion_penalty + phase_change)

if __name__ == "__main__":
    try:
        sumo_config_file = "../../projectnet.sumocfg"
        model_path = "DQL_models/with_controller_episode_100.h5"
        tester = TrafficLightRLTest(model_path)
        tester.test(sumo_config_file)
    except Exception as e:
        print(f"Error in main execution: {e}")
import json
import matplotlib.pyplot as plt

def plot_rewards(filename="rewards_history.json"):
    """Tracer les récompenses au fil des épisodes."""
    try:
        # Charger les données des récompenses depuis le fichier JSON
        with open(filename, 'r') as f:
            rewards_data = json.load(f)

        # Extraire les informations des épisodes et des récompenses
        episodes = [entry["episode"] for entry in rewards_data]
        rewards = [entry["reward"] for entry in rewards_data]

        # Tracer les récompenses
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, label="Récompense par épisode", color="blue", marker="o")
        plt.xlabel("Épisodes")
        plt.ylabel("Récompense")
        plt.title("Récompenses de l'agent au fil des épisodes")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Erreur lors du tracé des récompenses : {e}")

# Appeler la fonction pour afficher le graphique
plot_rewards()

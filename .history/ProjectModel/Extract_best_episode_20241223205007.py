import os
import re
import h5py
import matplotlib.pyplot as plt

# Dossier contenant les modèles
MODELS_DIR = "models"

def extract_performance_data():
    pattern = re.compile(r"traffic_light_control_episode_(\d+)\.h5")

    # Liste pour stocker les numéros d'épisodes et les performances
    episodes = []
    performances = []

    # Parcourir tous les fichiers dans le dossier des modèles
    for filename in os.listdir(MODELS_DIR):
        match = pattern.match(filename)
        if match:
            episode_number = int(match.group(1))  # Extraire le numéro d'épisode
            file_path = os.path.join(MODELS_DIR, filename)
            
            # Ouvrir le fichier .h5 pour extraire les données de performance
            with h5py.File(file_path, 'r') as f:
                # Par exemple, nous supposerons que la performance est stockée sous 'performance' dans chaque fichier
                if 'performance' in f:
                    performance = f['performance'][()]
                    episodes.append(episode_number)
                    performances.append(performance)

    return episodes, performances

def plot_performance(episodes, performances):
    # Tracer l'évolution de la performance
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, performances, marker='o', linestyle='-', color='b')
    plt.title("Évolution de la performance durant l'entraînement")
    plt.xlabel("Épisodes")
    plt.ylabel("Performance (par exemple, récompense)")
    plt.grid(True)
    plt.show()

def main():
    # Extraire les données de performance
    episodes, performances = extract_performance_data()

    if episodes and performances:
        # Tracer l'évolution de la performance
        plot_performance(episodes, performances)
    else:
        print("Aucune donnée de performance trouvée.")

if __name__ == "__main__":
    main()

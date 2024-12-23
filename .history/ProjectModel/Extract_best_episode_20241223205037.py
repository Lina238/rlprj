import os
import re

# Dossier contenant les modèles
MODELS_DIR = "models"

def extract_best_episode():
    pattern = re.compile(r"traffic_light_control_episode_(\d+)\.h5")

    best_episode = None
    best_filename = None

    # Parcourir tous les fichiers dans le dossier des modèles
    for filename in os.listdir(MODELS_DIR):
        match = pattern.match(filename)
        if match:
            episode_number = int(match.group(1))  # Extraire le numéro d'épisode
            if best_episode is None or episode_number > best_episode:
                best_episode = episode_number
                best_filename = filename

    if best_filename:
        print(f"Meilleur épisode trouvé : {best_filename}")
    else:
        print("Aucun fichier .h5 valide trouvé dans le dossier.")

if __name__ == "__main__":
    extract_best_episode()
#selon le code c'

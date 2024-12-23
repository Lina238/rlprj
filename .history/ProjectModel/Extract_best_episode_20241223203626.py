import os
import pickle

def extract_best_episode(episode_files_dir="episodes"):
    """Extrait et sauvegarde le meilleur épisode en fonction de la récompense."""
    best_episode = None
    best_reward = -float("inf")  # Initialisation avec une récompense très basse

    # Parcours des fichiers d'épisodes
    for episode_file in os.listdir(episode_files_dir):
        if episode_file.endswith(".pkl"):  # Vérification du format pickle
            episode_path = os.path.join(episode_files_dir, episode_file)
            with open(episode_path, "rb") as f:
                episode_data = pickle.load(f)

            # Calculer la récompense totale de l'épisode
            total_reward = sum([step["reward"] for step in episode_data])

            # Vérifier si cet épisode est meilleur
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode_data

    # Sauvegarder le meilleur épisode
    if best_episode:
        with open("best_episode.pkl", "wb") as f:
            pickle.dump(best_episode, f)
        print(f"Le meilleur épisode a été extrait et sauvegardé avec une récompense de {best_reward}.")
    else:
        print("Aucun épisode trouvé ou aucun épisode n'a de récompense valide.")

def main():
    extract_best_episode()  # Appel à la fonction d'extraction du meilleur épisode

if __name__ == "__main__":
    main()  # Exécution du programme

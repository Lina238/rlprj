import os
import re
import json

def extract_best_episode(models_dir):
    """
    Trouve le meilleur épisode parmi tous les fichiers .h5 dans le dossier `models_dir`.

    Args:
        models_dir (str): Le chemin vers le dossier contenant les fichiers .h5.
    
    Returns:
        dict: Les détails du meilleur épisode.
    """
    best_episode = None
    best_score = float('-inf')  # On suppose que le score plus élevé est meilleur.
    
    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(models_dir):
        if filename.endswith('.h5'):
            # Extraire le numéro d'épisode et/ou le score du nom de fichier
            match = re.search(r'episode_(\d+)_score_(\d+(\.\d+)?)', filename)
            if match:
                episode = int(match.group(1))
                score = float(match.group(2))
                
                # Comparer les scores pour trouver le meilleur
                if score > best_score:
                    best_score = score
                    best_episode = {"filename": filename, "episode": episode, "score": score}
    
    # Sauvegarder les détails du meilleur épisode dans un fichier JSON
    if best_episode:
        output_file = os.path.join(models_dir, 'extract_best_episode.json')
        with open(output_file, 'w') as f:
            json.dump(best_episode, f, indent=4)
        print(f"Meilleur épisode sauvegardé dans {output_file}.")
    else:
        print("Aucun fichier .h5 trouvé ou aucun fichier valide avec 'episode' et 'score'.")
    
    return best_episode

# Exemple d'utilisation
models_directory = "./models"
best = extract_best_episode(models_directory)
if best:
    print(f"Meilleur épisode : {best}")

import os
import tensorflow as tf

def extract_best_model(model_files_dir="models", output_file="best_model.h5"):
    """Extrait et sauvegarde le modèle le plus performant basé sur une évaluation."""
    best_model = None
    best_score = -float("inf")  # Initialisation avec un score très bas

    # Parcours des fichiers de modèles
    for model_file in os.listdir(model_files_dir):
        if model_file.endswith(".h5"):  # Vérification du format .h5
            model_path = os.path.join(model_files_dir, model_file)

            # Charger le modèle
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {model_file}: {e}")
                continue

            # Évaluer le modèle (ici, supposez une fonction spécifique pour calculer le score)
            # Par exemple, si vous avez des données de validation:
            # validation_data = ...
            # score = model.evaluate(validation_data)

            # Si vous n'avez pas de données de validation, définissez une métrique:
            score = calculate_model_score(model)

            # Vérifiez si ce modèle est meilleur
            if score > best_score:
                best_score = score
                best_model = model

    # Sauvegarder le meilleur modèle
    if best_model:
        best_model.save(output_file)
        print(f"Le meilleur modèle a été extrait et sauvegardé dans {output_file} avec un score de {best_score}.")
    else:
        print("Aucun modèle valide n'a été trouvé.")

def calculate_model_score(model):
    """
    Calcule un score pour le modèle.
    Remplacez cette fonction par une évaluation réelle basée sur vos données.
    """
    return model.count_params()  # Exemple: Plus le modèle a de paramètres, plus le score est élevé.

def main():
    extract_best_model()  # Appel à la fonction d'extraction du meilleur modèle

if __name__ == "__main__":
    main()  # Exécution du programme

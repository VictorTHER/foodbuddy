import tensorflow as tf
import pickle

def save_model_as_pickle(model_path, file_path):
    """
    Sauvegarde un modèle Keras/TensorFlow en format .pickle.

    Args:
        model (tf.keras.Model): Modèle à sauvegarder.
        file_path (str): Chemin où sauvegarder le fichier .pickle.
    """
    try:
        # Load the .h5 model
        model = tf.keras.models.load_model(model_path)

        try:
            # Sérialiser les poids et la structure
            model_data = {
                'architecture': model.to_json(),  # Architecture du modèle
                'weights': model.get_weights()   # Poids du modèle
            }

            # Sauvegarder avec pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Modèle sauvegardé en format pickle à : {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")
    except Exception as e:
        print(f"Erreur lors de la recherche du modèle .h5 : {e}")

def load_model_from_pickle(file_path):
    """
    Charge un modèle Keras/TensorFlow à partir d'un fichier .pickle.

    Args:
        file_path (str): Chemin du fichier .pickle.

    Returns:
        tf.keras.Model: Modèle reconstruit.
    """
    try:
        # Charger les données sérialisées
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        # Reconstruire le modèle
        model = tf.keras.models.model_from_json(model_data['architecture'])  # Reconstruire l'architecture
        model.set_weights(model_data['weights'])  # Charger les poids

        print(f"Modèle chargé depuis le fichier pickle : {file_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None


# Fonctions d'utilisation

# # Sauvegarde du modèle
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy')

save_model_as_pickle("RNN/MobileNet_Food101.h5", 'RNN/model.pickle')

# # Chargement du modèle
# loaded_model = load_model_from_pickle('model.pickle')
# loaded_model.summary()

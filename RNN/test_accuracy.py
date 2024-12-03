import tensorflow as tf

def get_model_accuracy(model, X_test, y_test):
    """
    Calcule l'accuracy d'un modèle Keras/TensorFlow sur un jeu de test.

    Args:
        model (tf.keras.Model): Le modèle à évaluer.
        X_test (numpy.ndarray): Les données de test (features).
        y_test (numpy.ndarray): Les labels de test.

    Returns:
        float: L'accuracy du modèle sur les données de test.
    """
    try:
        # Évaluation du modèle sur les données de test
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy du modèle sur les données de test : {accuracy * 100:.2f}%")
        return accuracy
    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
        return None

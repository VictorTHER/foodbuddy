import tensorflow as tf
import pickle
import os
h5_path = "foodbuddy/RNN/RNN.h5"
pickle_path = "foodbuddy/RNN/RNN.pkl"


def save_model_as_pickle():
    """
    Save a Keras/TensorFlow model as a pickle file.

    Saves the model architecture and weights for later reconstruction.
    """
    try:
        # Load the .h5 model
        model = tf.keras.models.load_model(h5_path)

        try:
            # Serialize the architecture and weights
            model_data = {
                'architecture': model.to_json(),  # Model architecture as JSON
                'weights': model.get_weights()   # Model weights
            }

            # Save using pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model successfully saved as pickle at: {pickle_path}")
        except Exception as e:
            print(f"Error during pickle saving: {e}")
    except Exception as e:
        print(f"Error loading .h5 model: {e}")


def load_RNN():
    """
    Load a Keras/TensorFlow model from a pickle or .h5 file.

    Priority:
    1. Attempt to load from pickle file.
    2. If pickle not found, attempt to load from .h5 file.

    Returns:
        tf.keras.Model: Reconstructed model, or None if loading fails.
    """
    # Attempt to load from pickle file
    if os.path.exists(pickle_path):
        print("found pickle file. Trying to load")
        try:
            with open(pickle_path, "rb") as f:
                model_data = pickle.load(f)

            # Reconstruct the model
            model = tf.keras.models.model_from_json(model_data['architecture'])
            model.set_weights(model_data['weights'])

            print("RNN Model loaded successfully from pickle.")
            return model

        except Exception as e:
            print(f"Error loading model from pickle: {e}")
            return None

    # If pickle loading fails, fallback to .h5 file
    print(f"Pickle file not found. Attempting to load .h5 file from '{h5_path}'.")

    if os.path.exists(h5_path):
        try:
            # Load the .h5 model directly
            model = tf.keras.models.load_model(h5_path)
            print("RNN Model loaded successfully from .h5.")
            return model
        except Exception as e:
            print(f"Error loading model from .h5: {e}")
            return None

    # If both loading methods fail
    print(f"Both pickle and .h5 files are missing or invalid.")
    return None

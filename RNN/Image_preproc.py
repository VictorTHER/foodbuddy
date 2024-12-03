from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

def convert_to_jpg(input_path, output_dir="converted_images"):
    """
    Convertit une image donnée en format JPG.

    Args:
        input_path (str): Chemin de l'image d'entrée.
        output_dir (str): Dossier où sauvegarder l'image JPG.

    Returns:
        str: Chemin de l'image convertie en JPG.
    """
    try:
        # Chargement de l'image avec PIL
        img = Image.open(input_path)

        # Vérifie et crée le dossier de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Chemin de sortie
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")

        # Conversion et sauvegarde
        rgb_img = img.convert("RGB")
        rgb_img.save(output_path, "JPEG")

        print(f"Image convertie et sauvegardée : {output_path}")
        return output_path
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")
        return None


def img_preprocessing(image_path, output_dir="preprocessed_images"):
    """
    Preprocessing de l'image pour l'adapter au modèle

    Args:
        image_path (str): Chemin de l'image à pré-traiter.
        output_dir (str): Dossier où sauvegarder les images prétraitées.

    Returns:
        str: Chemin de l'image prétraitée, prête pour le modèle.
    """
    try:
        # Chargement de l'image avec PIL
        img = Image.open(image_path)

        # Vérifie et crée le dossier de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Chemin de sortie
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")

        # Conversion en RGB si nécessaire
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Redimensionnement de l'image
        img_size = (224, 224)  # Taille standard pour les modèles CNN comme ResNet
        img = img.resize(img_size)

        # Sauvegarde de l'image en JPG
        img.save(output_path, "JPEG")

        # Optionnel : Prétraitement (normalisation)
        # Note : Normalisation se fait souvent après chargement en tant que tableau NumPy
        img_array = np.array(img) / 255.0  # Normalisation entre 0 et 1
        print(f"Image prétraitée et sauvegardée : {output_path}")

        return output_path
    except Exception as e:
        print(f"Erreur lors du preprocessing : {e}")
        return None


def augmentating_images(image_path, output_dir="augmented_images", num_augmented_images=5):
    """
    Augmentation de l'image input par l'utilisateur pour générer plusieurs variations.

    Args:
        image_path (str): Chemin de l'image à augmenter.
        output_dir (str): Dossier où sauvegarder les images augmentées.
        num_augmented_images (int): Nombre d'images augmentées à générer.

    Returns:
        list: Liste des chemins des images augmentées générées.
    """
    try:
        # Paramètres d'augmentation
        augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.2,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }

        # Charger l'image
        img = load_img(image_path)  # Chargement de l'image
        img_array = img_to_array(img)  # Conversion en tableau NumPy
        img_array = img_array.reshape((1,) + img_array.shape)  # Ajout d'une dimension pour ImageDataGenerator

        # Création de l'objet ImageDataGenerator
        datagen = ImageDataGenerator(**augmentation_params)

        # Vérifie et crée le dossier de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Génération d'images augmentées
        augmented_images_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir,
                                  save_prefix=f"{base_name}_aug", save_format="jpg"):
            i += 1
            augmented_images_paths.append(os.path.join(output_dir, f"{base_name}_aug_{i}.jpg"))
            if i >= num_augmented_images:  # Arrêter après le nombre d'images spécifié
                break

        print(f"{num_augmented_images} images augmentées générées dans {output_dir}.")
        return augmented_images_paths

    except Exception as e:
        print(f"Erreur lors de l'augmentation : {e}")
        return None

# Pipeline complète
def full_pipeline(image_path):
    """
    Exécute la pipeline complète : Conversion en JPG -> Préprocessing -> Augmentation.

    Args:
        image_path (str): Chemin de l'image d'entrée.

    Returns:
        list: Liste des chemins des images augmentées.
    """
    converted_path = convert_to_jpg(image_path)
    if converted_path:
        preprocessed_path = img_preprocessing(converted_path)
        if preprocessed_path:
            augmented_paths = augmentating_images(preprocessed_path)
            return augmented_paths
    return None

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def data_analysis(train_dir):
    # Liste des catégories
    categories = sorted(os.listdir(train_dir))
    categories

    # Création du dataframe
    df = pd.DataFrame(columns=['image_path', 'label'])
    dfs = []
    for category in categories:
        category_path = os.path.join(train_dir, category)
        image_data = []
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image_data.append({'image_path': image_path, 'label': category})
        df_category = pd.DataFrame(image_data)
        dfs.append(df_category)
    df = pd.concat(dfs, ignore_index=True)
    df

    # 1. Analyse des données
    # 1.1 Pondération
    df['label'].value_counts()

    #1.2 Visualisation
    plt.figure(figsize=(10, 6))
    df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel('Food Category')
    plt.ylabel('Count')
    plt.title('Value Counts of Food Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    #1.3 Class weights
    labels = df['label']
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weights_dict)
    print(class_weights) # Vérifier l'équipondération des données

    return df, categories

def preprocess_data(df, categories):
    # Train test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.shape
    val_df.shape

    # Détermination des paramètres de preproc
    # Image augmentation parameters
    augmentation_params = {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.2,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    # Image augmentation function
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        **augmentation_params
    )
    # Rescaling des images et des pixels
    img_size = (224, 224)
    batch_size = 32
    val_datagen = ImageDataGenerator(
        rescale = 1./255
        )

    # Preprocess des données train
    train_images = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label',

        target_size=img_size,
        batch_size=batch_size,

        shuffle=True,
        seed=42,
        class_mode='categorical'
    )

    # Preprocess des données val
    val_images = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label',

        target_size=img_size,
        batch_size=batch_size,

        shuffle=False,
        seed=42,
        class_mode='categorical'
    )

    # Visusalition intermédiaire avant encoding
    fig, axes = plt.subplots(6, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i, category in enumerate(categories):
        category_df = train_df[train_df['label'] == category]
        image_path = category_df.iloc[100]['image_path']
        img = plt.imread(image_path)
        axes[i].imshow(img)
        axes[i].set_title(category)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Encoding et revisualisation
    batch_train_images, batch_train_labels = next(train_images)
    class_names = np.argmax(batch_train_labels, axis=1)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(batch_train_images[i])
        plt.title(class_names[i])
        plt.axis('off')
    plt.show()

    return train_images, val_images

# Il faut input le chemin où les fichiers seront sauvegardés (file_path)

def model_activation(train_images, val_images, file_path = str, categories, monitoring_metric = 'accuracy', epoch_number = 50):
    # Détermination des paramètres
    pretrained_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False

    # x = Dense(32, activation='relu')(pretrained_model.output)
    # x = Dense(16, activation='relu')(x)
    outputs = Dense(len(categories), activation='softmax')(pretrained_model.output)
    model_mn = Model(pretrained_model.input, outputs, name='MobileNetV2')
    print(model_mn.summary())

    # Compiler le modèle
    model_mn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[monitoring_metric],
    )
    # Sauvegarde intermédiaire avec ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath= file_path,  # Nom de fichier pour la sauvegarde
        save_best_only=True,             # Sauvegarde uniquement le meilleur modèle
        save_weights_only=False,         # Sauvegarde le modèle complet (architecture + poids)
        monitor=monitoring_metric,              # Critère de surveillance
        mode='max',                      # Mode maximisation pour accuracy
        verbose=1                        # Affichage des logs
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor=monitoring_metric,
        patience=5,
        restore_best_weights=True
    )

    # Entraînement avec les callbacks
    history_mn = model_mn.fit(
        train_images,
        validation_data=val_images,
        epochs= epoch_number,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback  # Ajout du callback pour la sauvegarde
        ]
    )

    # Save le modèle
    model_mn.save('./RNN/MobileNet_Food101.h5')

    # Save le model en .pickle
    with open('fitted_model_rnn.pkl', 'wb') as f:
        pickle.dump(model_mn, f)
    print(f"Processed dataset saved at foodbuddy/KNN.")

    return history_mn, model_mn

def model_stats(val_images, history_mn):
    # Détermination des metrics de val
    val_loss_mn, val_acc_mn = model_mn.evaluate(val_images, verbose=0)
    print(val_loss_mn)
    print(val_acc_mn)

    # Comparaison des accuracy et loss des données val et train
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_mn.history['accuracy'])
    plt.plot(history_mn.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history_mn.history['loss'])
    plt.plot(history_mn.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

    return

def prediction_analysis(model_mn, val_images):
    # Making the prediction
    predictions = model_mn.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes

    true_classes = val_images.classes
    class_labels = list(val_images.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    plt.show()

    # KPIs finaux
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    return report, cm

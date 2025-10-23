#!/usr/bin/env python3
"""
train.py - Entraînement du modèle de classification de maladies de feuilles
Usage: python train.py <images_folder>
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
TARGET_ACCURACY = 0.90


def create_model(num_classes):
    """Crée un modèle basé sur MobileNetV2 (Transfer Learning)"""
    
    # Base pré-entraînée (ImageNet)
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze les couches de base
    base_model.trainable = False
    
    # Construction du modèle complet
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def organize_dataset(source_folder):
    """
    Organise le dataset en train/validation
    Retourne les chemins et les informations sur les classes
    """
    source = Path(source_folder)
    
    # Récupérer toutes les classes
    classes = []
    for subfolder in source.iterdir():
        if subfolder.is_dir():
            classes.append(subfolder.name)
    
    classes = sorted(classes)
    print(f"\n📊 Classes détectées: {len(classes)}")
    for i, cls in enumerate(classes):
        n_images = len(list((source / cls).glob('*.[jJ][pP][gG]'))) + \
                   len(list((source / cls).glob('*.[jJ][pP][eE][gG]'))) + \
                   len(list((source / cls).glob('*.[pP][nN][gG]')))
        print(f"   {i+1}. {cls}: {n_images} images")
    
    # Créer un dossier temporaire organisé
    temp_folder = Path("temp_dataset")
    train_dir = temp_folder / "train"
    val_dir = temp_folder / "validation"
    
    # Nettoyer si existe déjà
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Organiser les images
    print(f"\n🔄 Organisation du dataset (split {int((1-VALIDATION_SPLIT)*100)}/{int(VALIDATION_SPLIT*100)})...")
    
    for class_name in classes:
        # Créer les sous-dossiers
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Récupérer toutes les images
        class_folder = source / class_name
        images = list(class_folder.glob('*.[jJ][pP][gG]')) + \
                 list(class_folder.glob('*.[jJ][pP][eE][gG]')) + \
                 list(class_folder.glob('*.[pP][nN][gG]'))
        
        # Split train/validation
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=VALIDATION_SPLIT, 
            random_state=42
        )
        
        # Copier les images
        for img in train_imgs:
            shutil.copy(img, train_dir / class_name / img.name)
        
        for img in val_imgs:
            shutil.copy(img, val_dir / class_name / img.name)
        
        print(f"   ✓ {class_name}: {len(train_imgs)} train, {len(val_imgs)} validation")
    
    return train_dir, val_dir, classes


def create_data_generators(train_dir, val_dir):
    """Crée les générateurs de données avec augmentation"""
    
    # Augmentation pour le training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Pas d'augmentation pour la validation (juste normalisation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def plot_training_history(history, output_path):
    """Génère les graphiques de training"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Graphiques sauvegardés: {output_path}")


def save_model_package(model, classes, history, output_name="model_leaffliction"):
    """Sauvegarde le modèle + métadonnées dans un .zip"""
    
    output_folder = Path(output_name)
    output_folder.mkdir(exist_ok=True)
    
    # 1. Sauvegarder le modèle
    model.save(output_folder / "model.h5")
    print(f"💾 Modèle sauvegardé: {output_folder / 'model.h5'}")
    
    # 2. Sauvegarder les métadonnées
    metadata = {
        "classes": classes,
        "img_size": IMG_SIZE,
        "num_classes": len(classes),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "epochs": len(history.history['accuracy'])
    }
    
    with open(output_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📝 Métadonnées sauvegardées: {output_folder / 'metadata.json'}")
    
    # 3. Sauvegarder les graphiques
    plot_training_history(history, output_folder / "training_history.png")
    
    # 4. Créer le .zip
    shutil.make_archive(output_name, 'zip', output_folder)
    print(f"\n📦 Package créé: {output_name}.zip")
    
    # 5. Nettoyer le dossier temporaire
    shutil.rmtree(output_folder)
    
    return f"{output_name}.zip"


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <images_folder>")
        print("\nExemple:")
        print("  python train.py ./images/")
        sys.exit(1)
    
    source_folder = sys.argv[1]
    
    if not Path(source_folder).exists():
        print(f"❌ Erreur: Le dossier '{source_folder}' n'existe pas")
        sys.exit(1)
    
    print("=" * 70)
    print("🌿 LEAFFLICTION - TRAINING")
    print("=" * 70)
    
    # 1. Organisation du dataset
    train_dir, val_dir, classes = organize_dataset(source_folder)
    
    # 2. Création des générateurs
    print("\n🔧 Création des générateurs de données...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir)
    
    # 3. Création du modèle
    print(f"\n🧠 Création du modèle ({len(classes)} classes)...")
    model = create_model(len(classes))
    model.summary()
    
    # 4. Entraînement
    print("\n🚀 Début de l'entraînement...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Objectif validation accuracy: {TARGET_ACCURACY*100}%\n")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. Résultats finaux
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"   Train Accuracy:      {final_train_acc*100:.2f}%")
    print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
    
    if final_val_acc >= TARGET_ACCURACY:
        print(f"\n   ✅ Objectif atteint (>= {TARGET_ACCURACY*100}%)")
    else:
        print(f"\n   ⚠️  Objectif non atteint (< {TARGET_ACCURACY*100}%)")
        print("   💡 Conseils: augmentez EPOCHS ou ajoutez plus d'images")
    
    # 6. Sauvegarde
    print("\n💾 Sauvegarde du modèle...")
    output_name = Path(source_folder).name + "_model"
    zip_path = save_model_package(model, classes, history, output_name)
    
    # 7. Nettoyage
    print("\n🧹 Nettoyage des fichiers temporaires...")
    shutil.rmtree("temp_dataset")
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"\n📦 Fichier à soumettre: {zip_path}")
    print(f"📝 Pour tester: python predict.py {zip_path} <image_path>")
    print()


if __name__ == "__main__":
    main()
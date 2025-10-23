#!/usr/bin/env python3
"""
predict.py - Prédiction de maladie sur une image de feuille
Usage: python predict.py <model.zip> <image_path>
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2


def extract_model(zip_path):
    """Extrait le modèle du .zip"""
    extract_dir = Path("temp_model")
    
    # Nettoyer si existe
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    
    extract_dir.mkdir(exist_ok=True)
    
    # Extraire
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return extract_dir


def load_model_and_metadata(model_dir):
    """Charge le modèle et les métadonnées"""
    model_path = model_dir / "model.h5"
    metadata_path = model_dir / "metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées non trouvées: {metadata_path}")
    
    # Charger le modèle
    model = keras.models.load_model(model_path)
    
    # Charger les métadonnées
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def preprocess_image(img_path, img_size):
    """Prétraite l'image pour la prédiction"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def apply_transformation(img_path):
    """
    Applique une transformation simple (masque de la feuille)
    Similaire à Transformation.py mais simplifié
    """
    # Charger l'image avec OpenCV
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Seuillage pour isoler la feuille
    _, mask = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    
    # Appliquer le masque
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # Mettre un fond blanc
    white_bg = np.full_like(img_rgb, 255)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
    
    transformed = cv2.add(result, background)
    
    return img_rgb, transformed


def create_prediction_display(original, transformed, predicted_class, confidence):
    """
    Crée l'affichage final comme dans le PDF
    (2 images côte à côte + texte de prédiction)
    """
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    
    # Image originale
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(original)
    ax1.axis('off')
    ax1.set_title('Original', color='white', fontsize=14)
    
    # Image transformée
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(transformed)
    ax2.axis('off')
    ax2.set_title('Transformed', color='white', fontsize=14)
    
    # Titre principal
    fig.suptitle('=== DL classification ===', 
                 color='white', fontsize=18, y=0.95)
    
    # Texte de prédiction
    prediction_text = f'Class predicted : {predicted_class}'
    confidence_text = f'Confidence : {confidence:.2f}%'
    
    plt.figtext(0.5, 0.08, prediction_text, 
                ha='center', fontsize=16, color='#00FF00', weight='bold')
    plt.figtext(0.5, 0.02, confidence_text, 
                ha='center', fontsize=12, color='white')
    
    plt.tight_layout()
    return fig


def predict_image(model, metadata, img_path):
    """Fait la prédiction sur une image"""
    
    img_size = tuple(metadata['img_size'])
    classes = metadata['classes']
    
    # Prétraitement
    original_img, img_array = preprocess_image(img_path, img_size)
    
    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    predicted_class = classes[predicted_idx]
    
    return predicted_class, confidence


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model.zip> <image_path>")
        print("\nExemple:")
        print("  python predict.py images_model.zip ./test_image.jpg")
        sys.exit(1)
    
    model_zip = sys.argv[1]
    img_path = sys.argv[2]
    
    # Vérifications
    if not Path(model_zip).exists():
        print(f"❌ Erreur: Le fichier '{model_zip}' n'existe pas")
        sys.exit(1)
    
    if not Path(img_path).exists():
        print(f"❌ Erreur: L'image '{img_path}' n'existe pas")
        sys.exit(1)
    
    print("=" * 70)
    print("🌿 LEAFFLICTION - PREDICTION")
    print("=" * 70)
    
    # 1. Extraction du modèle
    print(f"\n📦 Extraction du modèle depuis {model_zip}...")
    model_dir = extract_model(model_zip)
    
    # 2. Chargement
    print("🧠 Chargement du modèle...")
    model, metadata = load_model_and_metadata(model_dir)
    
    print(f"   Classes disponibles: {len(metadata['classes'])}")
    print(f"   Accuracy du modèle: {metadata['final_val_accuracy']*100:.2f}%")
    
    # 3. Transformation de l'image
    print(f"\n🖼️  Traitement de l'image: {Path(img_path).name}")
    original, transformed = apply_transformation(img_path)
    
    # 4. Prédiction
    print("🔍 Prédiction en cours...")
    predicted_class, confidence = predict_image(model, metadata, img_path)
    
    print("\n" + "=" * 70)
    print("📊 RÉSULTAT")
    print("=" * 70)
    print(f"   Classe prédite : {predicted_class}")
    print(f"   Confiance      : {confidence:.2f}%")
    print("=" * 70)
    
    # 5. Affichage
    print("\n📊 Affichage du résultat...")
    fig = create_prediction_display(original, transformed, predicted_class, confidence)
    plt.show()
    
    # 6. Nettoyage
    shutil.rmtree(model_dir)
    
    print("\n✅ Prédiction terminée!\n")


if __name__ == "__main__":
    main()
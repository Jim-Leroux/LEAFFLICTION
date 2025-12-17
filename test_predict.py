#!/usr/bin/env python3
"""
test_predict.py - Script de test automatique pour predict.py
Usage: python test_predict.py <model.zip> <images_folder>
"""

import sys
import json
import zipfile
import shutil
from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Constants matching training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_model(zip_path):
    """Extrait le modèle du .zip"""
    extract_dir = Path("temp_model_test")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def load_model_and_metadata(model_dir):
    """Charge le modèle et les métadonnées"""
    model_path = model_dir / "model.pth"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées non trouvées: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    num_classes = metadata["num_classes"]

    # Rebuild the exact model structure
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, metadata


def preprocess_image(img_path, img_size):
    """Prétraite l'image pour la prédiction"""
    # Match the validation transform from train.py
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


def predict_image(model, metadata, img_path):
    """Fait la prédiction sur une image"""
    img_size = tuple(metadata["img_size"])
    classes = metadata["classes"]

    img_tensor = preprocess_image(img_path, img_size)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Get predictions as numpy for easier handling
    probs_numpy = probabilities.cpu().numpy()[0]

    predicted_idx = np.argmax(probs_numpy)
    confidence = probs_numpy[predicted_idx] * 100
    predicted_class = classes[predicted_idx]

    # Top 3 prédictions
    top_3_idx = np.argsort(probs_numpy)[-3:][::-1]
    top_3 = [(classes[i], probs_numpy[i] * 100) for i in top_3_idx]

    return predicted_class, confidence, top_3


def get_true_label_from_path(img_path):
    """Extrait le label réel depuis le chemin de l'image"""
    # Chemin typique : ./images/apple_healthy/image1.jpg
    parent = Path(img_path).parent.name
    return parent


def test_on_folder(model, metadata, images_folder, max_images=50, per_class=None):
    """Test le modèle sur un dossier d'images

    Args:
        model: Modèle entraîné
        metadata: Métadonnées du modèle
        images_folder: Dossier contenant les images
        max_images: Nombre total maximum d'images à tester
        per_class: Nombre maximum d'images par classe (None = illimité)
    """

    images_path = Path(images_folder)

    if not images_path.exists():
        print(f"❌ Le dossier '{images_folder}' n'existe pas")
        return

    # Récupérer toutes les images
    all_images = []

    # Parcourir les sous-dossiers (classes)
    for class_folder in images_path.iterdir():
        if not class_folder.is_dir():
            continue

        # Récupérer toutes les images de la classe
        class_images = (
            list(class_folder.glob("*.[jJ][pP][gG]"))
            + list(class_folder.glob("*.[jJ][pP][eE][gG]"))
            + list(class_folder.glob("*.[pP][nN][gG]"))
        )

        # Limiter par classe si spécifié
        if per_class is not None:
            class_images = class_images[:per_class]

        all_images.extend(class_images)

    # Limiter le nombre total d'images
    all_images = all_images[:max_images]

    if not all_images:
        print(f"❌ Aucune image trouvée dans '{images_folder}'")
        return

    print(f"\n📊 Test sur {len(all_images)} images...")
    print("=" * 80)

    # Statistiques
    correct = 0
    total = 0
    results = []

    # Tester chaque image
    for img_path in all_images:
        try:
            true_label = get_true_label_from_path(img_path)
            predicted_class, confidence, top_3 = predict_image(model, metadata, img_path)

            is_correct = predicted_class == true_label

            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"

            total += 1

            result = {
                "image": img_path.name,
                "true_label": true_label,
                "predicted": predicted_class,
                "confidence": confidence,
                "correct": is_correct,
                "top_3": top_3,
            }
            results.append(result)

            # Affichage ligne par ligne
            print(
                f"{status} {total:3d}. {img_path.name:30s} | "
                f"True: {true_label:20s} | "
                f"Pred: {predicted_class:20s} | "
                f"Conf: {confidence:5.1f}%"
            )

        except Exception as e:
            print(f"⚠️  Erreur avec {img_path.name}: {e}")

    # Résultats finaux
    accuracy = (correct / total * 100) if total > 0 else 0

    print("=" * 80)
    print(f"\n📊 RÉSULTATS GLOBAUX")
    print("=" * 80)
    print(f"   Total images testées : {total}")
    print(f"   Prédictions correctes : {correct}")
    print(f"   Prédictions incorrectes : {total - correct}")
    print(f"   Accuracy : {accuracy:.2f}%")

    # Objectif du projet
    if accuracy >= 90:
        print(f"\n   🎉 Objectif atteint ! (>= 90%)")
    else:
        print(f"\n   ⚠️  Objectif non atteint (< 90%)")

    # Afficher les erreurs
    errors = [r for r in results if not r["correct"]]

    if errors:
        print(f"\n❌ ERREURS DE PRÉDICTION ({len(errors)} erreurs)")
        print("=" * 80)

        for i, err in enumerate(errors[:10], 1):  # Limiter à 10 erreurs
            print(f"\n{i}. {err['image']}")
            print(f"   Vrai label    : {err['true_label']}")
            print(f"   Prédit        : {err['predicted']} ({err['confidence']:.1f}%)")
            print(f"   Top 3 prédictions :")
            for cls, conf in err["top_3"]:
                print(f"      - {cls:25s} : {conf:5.1f}%")

    # Matrice de confusion simplifiée
    print(f"\n📊 DISTRIBUTION PAR CLASSE")
    print("=" * 80)

    # Grouper par classe
    class_stats = {}
    for r in results:
        true_label = r["true_label"]
        if true_label not in class_stats:
            class_stats[true_label] = {"total": 0, "correct": 0}

        class_stats[true_label]["total"] += 1
        if r["correct"]:
            class_stats[true_label]["correct"] += 1

    # Afficher
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        cls_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {cls:25s} : {stats['correct']:2d}/{stats['total']:2d} ({cls_accuracy:5.1f}%)")

    print("=" * 80)

    return accuracy, results


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_predict.py <model.zip> <images_folder> [max_images] [per_class]")
        print("\nExemples:")
        print("  python test_predict.py images_model.zip ./images/")
        print("  python test_predict.py images_model.zip ./images/ 100")
        print("  python test_predict.py images_model.zip ./images/ 1000 200")
        print("  python test_predict.py images_model.zip ./images/ 1000 all")
        print("\nArguments:")
        print("  max_images  : Nombre total maximum d'images (défaut: 50)")
        print("  per_class   : Images par classe (défaut: 10, 'all' = illimité)")
        print("\nCe script teste le modèle sur plusieurs images automatiquement")
        sys.exit(1)

    model_zip = sys.argv[1]
    images_folder = sys.argv[2]
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    # Gestion du paramètre per_class
    if len(sys.argv) > 4:
        if sys.argv[4].lower() == "all":
            per_class = None  # Pas de limite par classe
        else:
            per_class = int(sys.argv[4])
    else:
        per_class = 10  # Par défaut: 10 images par classe

    # Vérifications
    if not Path(model_zip).exists():
        print(f"❌ Erreur: Le fichier '{model_zip}' n'existe pas")
        sys.exit(1)

    if not Path(images_folder).exists():
        print(f"❌ Erreur: Le dossier '{images_folder}' n'existe pas")
        sys.exit(1)

    print("=" * 80)
    print("🧪 TEST AUTOMATIQUE DE PREDICT.PY")
    print("=" * 80)
    print(f"   Max images total : {max_images}")
    print(f"   Max par classe   : {'Illimité' if per_class is None else per_class}")

    # Timer
    start_time = time.time()

    # 1. Extraction du modèle
    print(f"\n📦 Extraction du modèle depuis {model_zip}...")
    model_dir = extract_model(model_zip)

    # 2. Chargement
    print("🧠 Chargement du modèle...")
    model, metadata = load_model_and_metadata(model_dir)

    print(f"   Classes : {len(metadata['classes'])}")
    print(f"   Accuracy training : {metadata['final_train_accuracy']*100:.2f}%")
    print(f"   Accuracy validation : {metadata['final_val_accuracy']*100:.2f}%")

    # 3. Test sur les images
    accuracy, results = test_on_folder(model, metadata, images_folder, max_images, per_class)

    # 4. Temps d'exécution
    elapsed = time.time() - start_time
    print(f"\n⏱️  Temps d'exécution : {elapsed:.2f} secondes")
    print(f"   Temps moyen par image : {elapsed/len(results):.3f}s")

    # 5. Nettoyage
    shutil.rmtree(model_dir)

    print("\n✅ Test terminé !\n")


if __name__ == "__main__":
    main()

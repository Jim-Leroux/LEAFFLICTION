import sys
import json
import zipfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Constants matching training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_model(zip_path):
    extract_dir = Path("temp_model")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def load_model_and_metadata(model_dir):
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
    # No need to download weights, we allow loading ours
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, metadata


def preprocess_image(img_path, img_size):
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
    return img, img_tensor


def apply_transformation(img_path):
    # Visual transformation (OpenCV based, independent of ML framework)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Impossible de lire l'image avec OpenCV: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    white_bg = np.full_like(img_rgb, 255)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
    transformed = cv2.add(result, background)
    return img_rgb, transformed


def create_prediction_display(
    original, transformed, predicted_class, confidence
):
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("black")
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(original)
    ax1.axis("off")
    ax1.set_title("Original", color="white", fontsize=14)

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(transformed)
    ax2.axis("off")
    ax2.set_title("Transformed", color="white", fontsize=14)
    fig.suptitle(
        "=== DL classification ===", color="white", fontsize=18, y=0.95
    )

    prediction_text = f"Class predicted : {predicted_class}"
    confidence_text = f"Confidence : {confidence:.2f}%"
    plt.figtext(
        0.5,
        0.08,
        prediction_text,
        ha="center",
        fontsize=16,
        color="#00FF00",
        weight="bold",
    )
    plt.figtext(
        0.5, 0.02, confidence_text, ha="center", fontsize=12, color="white"
    )
    plt.tight_layout()
    return fig


def predict_image(model, metadata, img_path):
    img_size = tuple(metadata["img_size"])
    classes = metadata["classes"]

    _, img_tensor = preprocess_image(img_path, img_size)
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = classes[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    return predicted_class, confidence_pct


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model.zip> <image_path>")
        sys.exit(1)

    model_zip = sys.argv[1]
    img_path = sys.argv[2]

    if not Path(model_zip).exists():
        print(f"❌ Erreur: Le fichier '{model_zip}' n'existe pas")
        sys.exit(1)
    if not Path(img_path).exists():
        print(f"❌ Erreur: L'image '{img_path}' n'existe pas")
        sys.exit(1)

    print("=" * 70)
    print("🌿 LEAFFLICTION - PREDICTION")
    print("=" * 70)

    try:
        print(f"\n📦 Extraction du modèle depuis {model_zip}...")
        model_dir = extract_model(model_zip)

        print("🧠 Chargement du modèle...")
        model, metadata = load_model_and_metadata(model_dir)
        print(f"   Classes disponibles: {len(metadata['classes'])}")

        if "final_val_accuracy" in metadata:
            print(
                f"   Accuracy du modèle: "
                f"{metadata['final_val_accuracy']*100:.2f}%"
            )

        print(f"\n🖼️  Traitement de l'image: {Path(img_path).name}")
        # Use OpenCV for the visual "transformation" part as per original
        # script
        original_vis, transformed_vis = apply_transformation(img_path)

        print("🔍 Prédiction en cours...")
        predicted_class, confidence = predict_image(model, metadata, img_path)

        print("\n" + "=" * 70)
        print("📊 RÉSULTAT")
        print("=" * 70)
        print(f"   Classe prédite : {predicted_class}")
        print(f"   Confiance      : {confidence:.2f}%")
        print("=" * 70)

        print("\n📊 Affichage du résultat...")
        create_prediction_display(
            original_vis, transformed_vis, predicted_class, confidence
        )
        plt.show()

    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
    finally:
        if "model_dir" in locals() and model_dir.exists():
            shutil.rmtree(model_dir)

    print("\n✅ Prédiction terminée!\n")


if __name__ == "__main__":
    main()

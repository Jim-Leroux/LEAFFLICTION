import sys
import argparse
from pathlib import Path
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import cv2
import numpy as np


def transform_image(img_path):
    """Applique les 6 transformations avec PlantCV"""

    # Configuration PlantCV
    pcv.params.debug = None  # Désactive l'affichage debug

    # Lecture de l'image
    img, path, filename = pcv.readimage(str(img_path))

    # 1️⃣ Original
    original = img.copy()

    # 2️⃣ Gaussian Blur avec PlantCV
    gaussian_blur = pcv.gaussian_blur(img, ksize=(9, 9))

    # 3️⃣ Mask - Conversion en niveaux de gris et seuillage
    gray = pcv.rgb2gray(img)
    mask = pcv.threshold.binary(gray, threshold=135, object_type="light")

    # 4️⃣ ROI (apply mask)
    roi = pcv.apply_mask(img, mask, "white")

    # 5️⃣ Analyze Object (draw contours)
    # Trouve les objets dans le masque
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    analyze = img.copy()
    cv2.drawContours(analyze, contours, -1, (255, 0, 0), 2)

    # Alternative avec PlantCV (plus robuste)
    # analyze_obj = pcv.analyze_object(img, mask)

    # 6️⃣ Pseudolandmarks avec PlantCV
    pseudo = img.copy()

    # Utilise la fonction de pseudolandmarks de PlantCV
    if len(contours) > 0:
        # Trouve le plus grand contour (la feuille principale)
        largest_contour = max(contours, key=cv2.contourArea)
        obj_contour = [largest_contour]

        # Génère les pseudolandmarks (20 points par défaut)
        try:
            top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
                img, mask, obj_contour
            )
            left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
                img, mask, obj_contour
            )

            # Dessine les landmarks
            all_points = np.vstack([top, bottom, left, right])
            for point in all_points:
                if len(point) >= 2:
                    cv2.circle(
                        pseudo,
                        (int(point[0]), int(point[1])),
                        3,
                        (0, 255, 0),
                        -1,
                    )
        except BaseException:
            # Fallback si la méthode PlantCV échoue
            for i in range(
                0, len(largest_contour), max(1, len(largest_contour) // 20)
            ):
                x, y = largest_contour[i][0]
                cv2.circle(pseudo, (x, y), 3, (0, 255, 0), -1)

    return {
        "original": original,
        "gaussian_blur": gaussian_blur,
        "mask": mask,
        "roi": roi,
        "analyze_object": analyze,
        "pseudolandmarks": pseudo,
    }


def show_transformations(img_path):
    """Affiche les 6 transformations dans une grille"""
    transforms = transform_image(img_path)

    plt.figure(figsize=(12, 8))
    for i, (name, image) in enumerate(transforms.items(), 1):
        plt.subplot(2, 3, i)
        if name == "mask":
            plt.imshow(image, cmap="gray")
        else:
            # PlantCV utilise BGR, conversion en RGB pour affichage
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(name.replace("_", " ").title())

    plt.tight_layout()
    plt.show()


def save_transformations(src_file, dst_dir):
    """Sauvegarde toutes les transformations dans le dossier destination"""
    transforms = transform_image(src_file)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name, image in transforms.items():
        out_path = dst_dir / f"{src_file.stem}_{name}.png"

        # Sauvegarde avec OpenCV (PlantCV utilise le format BGR)
        if name == "mask":
            cv2.imwrite(str(out_path), image)
        else:
            cv2.imwrite(str(out_path), image)

        print(f"Sauvegardé: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Outil de transformation d'images de feuilles avec PlantCV"
    )
    parser.add_argument("-src", required=True, help="Image ou dossier source")
    parser.add_argument("-dst", help="Dossier destination (optionnel)")

    args = parser.parse_args()
    src = Path(args.src)

    if not src.exists():
        print(f"Erreur : le chemin '{src}' n'existe pas")
        sys.exit(1)

    if src.is_file():
        # Mode affichage pour une seule image
        print(f"Traitement de l'image : {src.name}")
        show_transformations(src)

    elif src.is_dir():
        # Mode batch pour un dossier
        dst = Path(args.dst) if args.dst else Path("augmented_directory")

        image_files = (
            list(src.glob("*.jpg"))
            + list(src.glob("*.jpeg"))
            + list(src.glob("*.png"))
            + list(src.glob("*.JPG"))
            + list(src.glob("*.PNG"))
        )

        if not image_files:
            print(f"Aucune image trouvée dans {src}")
            sys.exit(1)

        print(f"Traitement de {len(image_files)} image(s)...")

        for file in image_files:
            print(f"\nTraitement : {file.name}")
            try:
                save_transformations(file, dst)
            except Exception as e:
                print(f"Erreur avec {file.name}: {e}")

        print(f"\n✓ Toutes les transformations sont dans : {dst}")

    else:
        print("Erreur : chemin invalide")
        sys.exit(1)


if __name__ == "__main__":
    main()


# commande de test :

# python Transformation.py -src ./images/Apple_rust/image\ \(1\).JPG

# python Transformation.py -src ./images/Apple_rust/ -dst augmented_directory

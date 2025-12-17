import os
import sys

from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path


def increase_by_folder(folder):
    images = Path(folder)
    out_root = Path("augmented_directory")
    out_root.mkdir(exist_ok=True)

    # Boucle sur chaque sous-dossier
    for subfolder in images.iterdir():
        if subfolder.is_dir():
            out_dir = out_root / subfolder.name
            out_dir.mkdir(exist_ok=True)

            for element in subfolder.iterdir():
                if element.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(element)
                    name = element.stem
                    ext = element.suffix

                    # 1. Rotation
                    img.rotate(45).save(out_dir / f"{name}_rotated{ext}")

                    # 2. Blur
                    img.filter(ImageFilter.GaussianBlur(radius=5)).save(out_dir / f"{name}_blurred{ext}")

                    # 3. Contrast
                    ImageEnhance.Contrast(img).enhance(1.5).save(out_dir / f"{name}_contrast{ext}")

                    # 4. Zoom (x2)
                    w, h = img.size
                    crop = img.crop((w / 4, h / 4, w * 3 / 4, h * 3 / 4)).resize((w, h))
                    crop.save(out_dir / f"{name}_zoom{ext}")

                    # 5. Illumination
                    ImageEnhance.Brightness(img).enhance(1.8).save(out_dir / f"{name}_bright{ext}")

                    # 6. Projective
                    coeffs = (1, 0.2, -100, 0.2, 1, -50, 0.001, 0.001)
                    img.transform((w, h), Image.PERSPECTIVE, coeffs).save(out_dir / f"{name}_projective{ext}")

    print(f"Toutes les images ont été augmentées dans : {out_root}")


def increase_by_file(file):

    os.makedirs("by_file", exist_ok=True)

    file = Path(file)
    img = Image.open(file)

    f_name = file.name.split(".")
    name = f_name[0]
    extension = f_name[1]

    print(file.name)

    # 1. Rotation (ici 45°)
    rotated = img.rotate(45)
    rotated.save(f"by_file/{name}_rotated.{extension}")

    # 2. Blur (flou gaussien)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
    blurred.save(f"by_file/{name}_blurred.{extension}")

    # 3. Contrast (x1.9)
    contrast = ImageEnhance.Contrast(img).enhance(1.9)
    contrast.save(f"by_file/{name}_contrast.{extension}")

    # 4. Scaling (Zoom en x2)
    w, h = img.size

    # définir la zone de zoom (ici zoom 2x sur le centre)
    left = w / 4
    top = h / 4
    right = w * 3 / 4
    bottom = h * 3 / 4

    zoomed = img.crop((left, top, right, bottom))
    zoomed = zoomed.resize((w, h))  # remettre la taille originale
    zoomed.save(f"by_file/{name}_zoom.{extension}")

    # 5. Illumination (luminosité x1.8)
    bright = ImageEnhance.Brightness(img).enhance(1.8)
    bright.save(f"by_file/{name}_bright.{extension}")

    # 6. Projective transform (perspective)
    # on définit les coefficients pour transformer
    coeffs = (1, 0.2, -100, 0.2, 1, -50, 0.001, 0.001)  # ligne 1  # ligne 2  # ligne 3 (courbure perspective)
    projective = img.transform((img.width, img.height), Image.PERSPECTIVE, coeffs)
    projective.save(f"by_file/{name}_projective.{extension}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Augmentation.py <images_folder>")
        sys.exit()

    increase_by_file(sys.argv[1])

    # increase_by_folder(sys.argv[1])

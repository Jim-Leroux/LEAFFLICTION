import os
import sys

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path

os.makedirs("images/by_file", exist_ok=True)

def increase_by_folder(folder):
    folder_name = folder.split("/")[1]

    os.makedirs(f"images/{folder_name}", exist_ok=True)

    images = Path(folder)

    for element in images.iterdir():

        f_name = element.name.split(".")
        name = f_name[0]
        extension = f_name[1]
        img = Image.open(element)

        # 1. Rotation (ici 45°)
        rotated = img.rotate(45)
        rotated.save(f"images/{folder_name}/{name}_rotated.{extension}")

        # 2. Blur (flou gaussien)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
        blurred.save(f"images/{folder_name}/{name}_blurred.{extension}")

        # 3. Contrast (x1.5)
        contrast = ImageEnhance.Contrast(img).enhance(1.5)
        contrast.save(f"images/{folder_name}/{name}_contrast.{extension}")

        # 4. Scaling (redimensionner en 200x200)
        scaled = img.resize((200, 200))
        scaled.save(f"images/{folder_name}/{name}_scaled.{extension}")

        # 5. Illumination (luminosité x1.8)
        bright = ImageEnhance.Brightness(img).enhance(1.8)
        bright.save(f"images/{folder_name}/{name}_bright.{extension}")

        # 6. Projective transform (perspective)
        # on définit les coefficients pour transformer
        coeffs = (1, 0.2, -100,   # ligne 1
                0.2, 1, -50,    # ligne 2
                0.001, 0.001)   # ligne 3 (courbure perspective)
        projective = img.transform((img.width, img.height), Image.PERSPECTIVE, coeffs)
        projective.save(f"images/{folder_name}/{name}_projective.{extension}")

def increase_by_file(file):
    file = Path(file)
    img = Image.open(file)

    f_name = file.name.split(".")
    name = f_name[0]
    extension = f_name[1]

    print(file.name)

    # 1. Rotation (ici 45°)
    rotated = img.rotate(45)
    rotated.save(f"images/by_file/{name}_rotated.{extension}")

    # 2. Blur (flou gaussien)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
    blurred.save(f"images/by_file/{name}_blurred.{extension}")

    # 3. Contrast (x1.9)
    contrast = ImageEnhance.Contrast(img).enhance(1.9)
    contrast.save(f"images/by_file/{name}_contrast.{extension}")

    # 4. Scaling (Zoom en x2)
    w, h = img.size

    # définir la zone de zoom (ici zoom 2x sur le centre)
    left = w/4
    top = h/4
    right = w*3/4
    bottom = h*3/4

    zoomed = img.crop((left, top, right, bottom))
    zoomed = zoomed.resize((w, h))  # remettre la taille originale
    zoomed.save(f"images/by_file/{name}_zoom.{extension}")

    # 5. Illumination (luminosité x1.8)
    bright = ImageEnhance.Brightness(img).enhance(1.8)
    bright.save(f"images/by_file/{name}_bright.{extension}")

    # 6. Projective transform (perspective)
    # on définit les coefficients pour transformer
    coeffs = (1, 0.2, -100,   # ligne 1
            0.2, 1, -50,    # ligne 2
            0.001, 0.001)   # ligne 3 (courbure perspective)
    projective = img.transform((img.width, img.height), Image.PERSPECTIVE, coeffs)
    projective.save(f"images/by_file/{name}_projective.{extension}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Augmentation.py <images_folder>")
        sys.exit()

    # increase_by_folder(sys.argv[1])

    increase_by_file(sys.argv[1])

    folder = sys.argv[1].split("/")[1]

    print(folder)


# REPRENDRE UN NOUVEAU DOSSIER IMAGES PROPRES.
# RETOURNER LE DATA SET DANS UN DOSSIER AUGMENTED_DIRECTORY
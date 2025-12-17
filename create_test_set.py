#!/usr/bin/env python3
"""
create_test_set.py - Extrait 40 images par classe pour créer un test set isolé
Usage: python create_test_set.py <source_folder>
"""

import sys
import shutil
from pathlib import Path
import random


def create_test_set(source_folder, n_images=40):
    """
    Extrait n_images par classe pour créer un test set
    Retire ces images du dataset source
    """
    source = Path(source_folder)

    if not source.exists():
        print(f"❌ Erreur: '{source}' n'existe pas")
        sys.exit(1)

    # Dossiers de sortie
    test_set_dir = Path("test_set")
    remaining_dir = Path("dataset_for_training")

    # Nettoyer si existe
    if test_set_dir.exists():
        shutil.rmtree(test_set_dir)
    if remaining_dir.exists():
        shutil.rmtree(remaining_dir)

    test_set_dir.mkdir(exist_ok=True)
    remaining_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("🔬 CRÉATION DU TEST SET")
    print("=" * 70)
    print(f"📂 Source: {source}")
    print(f"🎯 Images par classe: {n_images}\n")

    total_test = 0
    total_remaining = 0

    # Parcourir chaque classe (sous-dossier)
    for class_folder in sorted(source.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        print(f"📁 Classe: {class_name}")

        # Récupérer toutes les images
        extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
        images = []
        for ext in extensions:
            images.extend(list(class_folder.glob(ext)))

        total_images = len(images)

        if total_images == 0:
            print(f"   ⚠️  Aucune image trouvée, ignoré\n")
            continue

        if total_images < n_images:
            print(f"   ⚠️  Seulement {total_images} images disponibles")
            n_to_extract = total_images
        else:
            n_to_extract = n_images

        # Mélanger aléatoirement (seed fixe pour reproductibilité)
        random.seed(42)
        random.shuffle(images)

        # Séparer
        test_images = images[:n_to_extract]
        remaining_images = images[n_to_extract:]

        # Créer les sous-dossiers
        test_class_dir = test_set_dir / class_name
        remaining_class_dir = remaining_dir / class_name

        test_class_dir.mkdir(parents=True, exist_ok=True)
        remaining_class_dir.mkdir(parents=True, exist_ok=True)

        # Copier les images
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)

        for img in remaining_images:
            shutil.copy2(img, remaining_class_dir / img.name)

        total_test += len(test_images)
        total_remaining += len(remaining_images)

        print(f"   ✅ Test: {len(test_images):4d} | Training: {len(remaining_images):4d}")

    print("\n" + "=" * 70)
    print("✅ SÉPARATION TERMINÉE")
    print("=" * 70)
    print(f"\n📊 Résumé:")
    print(f"   • test_set/               : {total_test} images au total")
    print(f"   • dataset_for_training/   : {total_remaining} images au total")

    print(f"\n💡 Prochaines étapes:")
    print(f"   1️⃣  python Augmentation.py -r dataset_for_training/")
    print(f"   2️⃣  python train.py augmented_directory/")
    print(f"   3️⃣  python predict.py model.zip test_set/<classe>/<image>.jpg")

    print(f"\n⚠️  IMPORTANT:")
    print(f"   Le dossier 'test_set/' ne doit JAMAIS être utilisé pour l'entraînement!")
    print(f"   Ces images sont pour tester le modèle final uniquement.\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_test_set.py <source_folder> [n_images]")
        print("\nExemples:")
        print("  python create_test_set.py ./PlantVillage/Apple/")
        print("  python create_test_set.py ./PlantVillage/Apple/ 50")
        print("\nPar défaut: 40 images par classe")
        sys.exit(1)

    source_folder = sys.argv[1]
    n_images = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    create_test_set(source_folder, n_images)


if __name__ == "__main__":
    main()

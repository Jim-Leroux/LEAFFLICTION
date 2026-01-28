import sys

from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance

import shutil


def augment_images(file, sub_directory_path):
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        with Image.open(file) as img:
            img.load()
            name = file.stem
            ext = file.suffix
            w, h = img.size

        shutil.copy(file, sub_directory_path / f"{name}{ext}")

        img.rotate(45).save(sub_directory_path / f"{name}_rotated{ext}")

        img.filter(ImageFilter.GaussianBlur(radius=5)).save(
            sub_directory_path / f"{name}_blurred{ext}"
        )

        ImageEnhance.Contrast(img).enhance(1.5).save(
            sub_directory_path / f"{name}_contrast{ext}"
        )

        w, h = img.size
        crop = img.crop(
            (w / 4, h / 4, w * 3 / 4, h * 3 / 4)
        ).resize((w, h))
        crop.save(sub_directory_path / f"{name}_zoom{ext}")

        ImageEnhance.Brightness(img).enhance(1.8).save(
            sub_directory_path / f"{name}_bright{ext}"
        )

        coeffs = (1, 0.2, -100, 0.2, 1, -50, 0.001, 0.001)
        img.transform((w, h), Image.PERSPECTIVE, coeffs).save(
            sub_directory_path / f"{name}_projective{ext}"
        )


def augment_by_folder(path_to_folder, destination_path):
    folders = [d for d in path_to_folder.iterdir() if d.is_dir()]

    max_img = max(len(list(d.glob('*'))) for d in folders)

    for directory in folders:
        sub_dst = destination_path / directory.name
        sub_dst.mkdir(exist_ok=True)
        ext = [".jpg", ".jpeg", ".png"]
        images = [f for f in directory.iterdir() if f.suffix.lower() in ext]

        for img_path in images:
            shutil.copy(img_path, sub_dst)

        current_count = len(images)
        img_index = 0

        while current_count < max_img:
            source_file = images[img_index % len(images)]
            augment_images(source_file, sub_dst)
            current_count += 6
            img_index += 1


def augment_by_file(path_to_file):
    destination_path = Path("augmented_directory_by_file")

    Path.mkdir(destination_path, exist_ok=True)
    augment_images(path_to_file, destination_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python Augmentation.py <path to folder>")

    path_to_folder = Path(sys.argv[1])

    if not path_to_folder.is_file():
        sys.exit("The path does not refer to a file")

    augment_by_file(path_to_folder)

    # if not path_to_folder.is_dir():
    #     sys.exit("The path does not refer to a folder")

    # destination_path = Path("augmented_directory")

    # Path.mkdir(destination_path, exist_ok=True)

    # augment_by_folder(path_to_folder, destination_path.resolve())


# By file cmd : python ex02.py images/Apple_Black_rot/image\ \(1\).JPG

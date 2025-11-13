#!/usr/bin/env python3
"""
Augmentation.py - Data augmentation for plant disease dataset.

This program applies 6 types of augmentation to balance the dataset.
It can work on a single image or recursively on entire directories.
"""

import sys
from PIL import Image, ImageFilter
from pathlib import Path
import argparse


def apply_augmentations(img, output_dir, base_name, extension):
    """
    Apply 6 types of augmentation to an image and save them.

    Args:
        img (PIL.Image): Source image
        output_dir (Path): Directory to save augmented images
        base_name (str): Base name for output files
        extension (str): File extension (e.g., '.jpg')

    Returns:
        list: List of created file paths
    """
    w, h = img.size
    created_files = []

    # 1. Flip (horizontal flip)
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_path = output_dir / f"{base_name}_Flip{extension}"
    flipped.save(flip_path)
    created_files.append(flip_path)

    # 2. Rotate (45 degrees)
    rotated = img.rotate(45, expand=False)
    rotate_path = output_dir / f"{base_name}_Rotate{extension}"
    rotated.save(rotate_path)
    created_files.append(rotate_path)

    # 3. Skew (shear transformation using affine)
    # Approximate skew using transform
    skewed = img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0.3, 0, 0, 1, 0),
        Image.BICUBIC
    )
    skew_path = output_dir / f"{base_name}_Skew{extension}"
    skewed.save(skew_path)
    created_files.append(skew_path)

    # 4. Shear (vertical shear)
    sheared = img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, 0.3, 1, 0),
        Image.BICUBIC
    )
    shear_path = output_dir / f"{base_name}_Shear{extension}"
    sheared.save(shear_path)
    created_files.append(shear_path)

    # 5. Crop (zoom on center)
    left, top = w // 4, h // 4
    right, bottom = w * 3 // 4, h * 3 // 4
    cropped = img.crop((left, top, right, bottom)).resize((w, h))
    crop_path = output_dir / f"{base_name}_Crop{extension}"
    cropped.save(crop_path)
    created_files.append(crop_path)

    # 6. Distortion (perspective transform + blur)
    coeffs = (1, 0.2, -50, 0.2, 1, -30, 0.001, 0.001)
    distorted = img.transform(
        (w, h),
        Image.PERSPECTIVE,
        coeffs
    ).filter(ImageFilter.GaussianBlur(radius=2))
    distortion_path = output_dir / f"{base_name}_Distortion{extension}"
    distorted.save(distortion_path)
    created_files.append(distortion_path)

    return created_files


def augment_single_image(image_path):
    """
    Augment a single image and display the results.

    Args:
        image_path (str or Path): Path to the image file
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    if not image_path.is_file():
        print(f"Error: '{image_path}' is not a file.")
        sys.exit(1)

    # Create output directory
    output_dir = Path("augmented_images")
    output_dir.mkdir(exist_ok=True)

    # Load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Get filename components
    base_name = image_path.stem
    extension = image_path.suffix

    print(f"Processing: {image_path.name}")
    print("Applying 6 augmentations...")

    # Apply augmentations
    created_files = apply_augmentations(
        img,
        output_dir,
        base_name,
        extension
    )

    print(f"\nAugmented images saved in: {output_dir}/")
    for file_path in created_files:
        print(f"  - {file_path.name}")


def count_images_in_dir(directory):
    """
    Count image files in a directory.

    Args:
        directory (Path): Directory to scan

    Returns:
        int: Number of image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            count += 1
    return count


def find_image_directories(root_path):
    """
    Find all directories containing images, grouped by parent.

    Args:
        root_path (Path): Root directory to search

    Returns:
        dict: Dictionary with parent dirs as keys and
              list of (subdir, image_count) as values
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    groups = {}

    # Check if root_path itself contains images
    has_images = any(
        f.suffix.lower() in image_extensions
        for f in root_path.iterdir()
        if f.is_file()
    )

    if has_images:
        # Root contains images directly - treat it as single group
        groups[root_path.name] = [(root_path, count_images_in_dir(root_path))]
        return groups

    # Scan subdirectories
    for item in root_path.iterdir():
        if not item.is_dir():
            continue

        # Check if this subdirectory contains images
        subdir_has_images = any(
            f.suffix.lower() in image_extensions
            for f in item.iterdir()
            if f.is_file()
        )

        if subdir_has_images:
            # This is a leaf directory with images
            parent_name = root_path.name
            if parent_name not in groups:
                groups[parent_name] = []
            groups[parent_name].append((item, count_images_in_dir(item)))
        else:
            # Recurse deeper
            sub_groups = find_image_directories(item)
            for key, value in sub_groups.items():
                if key not in groups:
                    groups[key] = []
                groups[key].extend(value)

    return groups


def balance_dataset(root_path, output_name='augmented_directory'):
    """
    Balance the dataset by augmenting underrepresented classes.

    Args:
        root_path (Path): Root directory containing subdirectories
                         with images
        output_name (str): Name of the output directory
    """
    root_path = Path(root_path)

    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist.")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"Error: '{root_path}' is not a directory.")
        sys.exit(1)

    # Create output directory
    output_root = Path(output_name)
    output_root.mkdir(exist_ok=True)

    print(f"Analyzing dataset structure in: {root_path}")
    print("-" * 60)

    # Find all image directories grouped by parent
    groups = find_image_directories(root_path)

    if not groups:
        print("No image directories found.")
        return

    # Process each group independently
    for group_name, directories in groups.items():
        print(f"\nGroup: {group_name}")
        print("=" * 60)

        # Find max images in this group
        max_images = max(count for _, count in directories)
        print(f"Target: {max_images} images per category\n")

        # Process each directory in the group
        for directory, current_count in directories:
            relative_path = directory.relative_to(root_path)
            output_dir = output_root / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"  {directory.name}: {current_count} images", end="")

            # Copy original images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            original_images = [
                f for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            for img_file in original_images:
                output_path = output_dir / img_file.name
                if not output_path.exists():
                    img = Image.open(img_file)
                    img.save(output_path)

            # Calculate how many augmentations needed
            images_needed = max_images - current_count

            if images_needed <= 0:
                print(" -> Already balanced [OK]")
                continue

            print(f" -> Need {images_needed} more images")

            # Generate augmentations cyclically
            full_cycles = images_needed // len(original_images)
            remaining = images_needed % len(original_images)

            generated_count = 0

            # Full cycles: apply all 6 augmentations
            if full_cycles > 0:
                for img_file in original_images:
                    img = Image.open(img_file)
                    base_name = img_file.stem
                    extension = img_file.suffix

                    created = apply_augmentations(
                        img,
                        output_dir,
                        base_name,
                        extension
                    )
                    generated_count += len(created)

                    if generated_count >= images_needed:
                        break

            # Remaining images: apply augmentations to first N images
            if generated_count < images_needed:
                for img_file in original_images[:remaining]:
                    img = Image.open(img_file)
                    base_name = img_file.stem
                    extension = img_file.suffix

                    created = apply_augmentations(
                        img,
                        output_dir,
                        base_name,
                        extension
                    )
                    generated_count += min(
                        len(created),
                        images_needed - generated_count
                    )

                    if generated_count >= images_needed:
                        break

            print(f"     Generated {generated_count} augmented images [OK]")

    print("\n" + "=" * 60)
    print("Dataset balanced successfully!")
    print(f"Output directory: {output_root.absolute()}")


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Augment images to balance plant disease dataset"
    )
    parser.add_argument(
        'path',
        help='Path to image file or directory'
    )
    parser.add_argument(
        '-r', '--repository',
        action='store_true',
        help='Process entire repository and balance dataset'
    )
    parser.add_argument(
        '-o', '--output',
        default='augmented_directory',
        help='Output directory name (default: augmented_directory)'
    )

    args = parser.parse_args()

    if args.repository:
        # Repository mode: balance entire dataset
        balance_dataset(args.path, args.output)
    else:
        # Single image mode
        augment_single_image(args.path)


if __name__ == "__main__":
    main()
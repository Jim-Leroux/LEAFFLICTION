"""
Distribution.py - Analyze and visualize plant disease dataset distribution.

This program takes a directory containing plant disease images organized in
subdirectories and generates pie charts and bar charts showing the distribution
of images across different disease categories.
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path


def get_data(images_dir):
    """
    Extract dataset information from directory structure.

    Args:
        images_dir (Path): Path to the directory containing subdirectories
                          with images

    Returns:
        dict: Dictionary with case names as keys and metadata as values
    """
    case_dict = {}

    # For each subdirectory
    for element in images_dir.iterdir():
        if not element.is_dir():
            continue

        case_name = element.name

        # Split name to extract species and state
        parts = element.name.split("_", 1)
        species, state = parts if len(parts) == 2 else (parts[0], "unknown")

        # Count image files in subdirectory
        img_files = [f for f in element.iterdir() if f.is_file()]

        case_dict[case_name] = {
            "species": species,
            "state": state,
            "path": element,
            "images_nb": len(img_files),
        }

    return case_dict


def display_charts(case_dict, parent_dir_name):
    """
    Display pie chart and bar chart for the dataset distribution.

    Args:
        case_dict (dict): Dictionary containing dataset information
        parent_dir_name (str): Name of the parent directory for chart titles
    """
    if not case_dict:
        print("No subdirectories found in the provided directory.")
        return

    # Extract labels and counts
    labels = []
    counts = []
    for case, infos in case_dict.items():
        labels.append(case)
        counts.append(infos['images_nb'])

    print("Labels:", labels)
    print("Counts:", counts)

    # Generate colors
    colors = plt.cm.tab20.colors[:len(labels)]

    # --- Pie Chart ---
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(f"Répartition des images - {parent_dir_name}")
    plt.show()

    # --- Bar Chart ---
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nombre d'images")
    plt.xlabel("Catégories")
    plt.title(f"Histogramme du nombre d'images - {parent_dir_name}")
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the distribution analysis."""
    if len(sys.argv) < 2:
        print("Usage: python Distribution.py <images_folder>")
        sys.exit(1)

    images_path = Path(sys.argv[1])

    # Check if directory exists
    if not images_path.exists():
        print(f"Error: Directory '{images_path}' does not exist.")
        sys.exit(1)

    if not images_path.is_dir():
        print(f"Error: '{images_path}' is not a directory.")
        sys.exit(1)

    # Get parent directory name for chart titles
    parent_dir_name = images_path.name

    # Extract data from directory structure
    case_dict = get_data(images_path)

    # Display visualizations
    display_charts(case_dict, parent_dir_name)


if __name__ == "__main__":
    main()
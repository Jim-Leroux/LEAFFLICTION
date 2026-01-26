import sys

from pathlib import Path

import matplotlib.pyplot as plt

data = {}

def images_counter(dir):
    number_of_images = 0
    for files in dir.iterdir():
        if files.is_file() and files.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            number_of_images += 1
    return (number_of_images)

def get_data(path_to_folder):
    for dir in path_to_folder.iterdir():
        if dir.is_dir():
            data[dir.name] = images_counter(dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python Distribution.py <path to folder>")

    path_to_folder = Path(sys.argv[1])

    if not path_to_folder.is_dir():
        sys.exit("The path does not refer to a folder.")

    get_data(path_to_folder)

    if not data:
        sys.exit("The folder appears to be empty.")

    colors = plt.cm.tab20.colors[: len(data.keys())]

    # --- Pie Charts ---
    plt.figure(figsize=(10, 10))
    plt.pie(data.values(), labels=data.keys(), autopct="%1.1f%%", colors=colors)
    plt.title("Répartition des images par plante et état")
    plt.show()

    # --- Bar Charts ---
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values(), color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Nombre d'images")
    plt.title("Histogramme du nombre d'images par cas")
    plt.tight_layout()
    plt.show()

    sys.exit(1)
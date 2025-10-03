import matplotlib.pyplot as plt
import sys

from pathlib import Path

labels = []
counts = []


def get_data(images):
    case_dict = {}

    # For each repository :
    for element in images.iterdir():
        if not element.is_dir():
            continue

        case_name = element.name

        # Split name and state
        parts = element.name.split("_", 1)
        species, state = parts if len(parts) == 2 else (parts[0], "unknown")
            
        # List each images
        img_files = [f for f in element.iterdir() if f.is_file()]

        if case_name not in case_dict:
            case_dict[case_name] = {}
        case_dict[case_name] = {\
            "species":species,
            "state":state,
            "path": element,
            "images_nb": len(img_files),\
            # "files": img_files
        }

    return case_dict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Distribution.py <images_folder>")
        sys.exit(1)

    images = Path(sys.argv[1])

    case_dict = get_data(images)

    for case, infos in case_dict.items():
        labels.append(case)
        counts.append(infos['images_nb'])

    print(labels)
    print(counts)

    colors = plt.cm.tab20.colors[:len(labels)]

    # --- Pie Charts ---
    plt.figure(figsize=(10,10))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title("Répartition des images par plante et état")
    plt.show()

    # --- Bar Charts ---
    plt.figure(figsize=(10,5))
    plt.bar(labels, counts, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Nombre d'images")
    plt.title("Histogramme du nombre d'images par cas")
    plt.tight_layout()
    plt.show()
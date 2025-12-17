import sys
import json
import shutil
import hashlib
import time
import copy
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
TARGET_ACCURACY = 0.90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_time(seconds):
    """Format time in seconds to a readable string"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def check_device():
    print(f"🖥️  Mode de calcul: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("   ⚠️  Utilisation du CPU. L'entraînement sera plus lent.")


def organize_dataset(source_folder):
    source = Path(source_folder)
    classes = [d.name for d in source.iterdir() if d.is_dir()]
    classes.sort()

    print(f"\n📊 Classes détectées: {len(classes)}")
    total_images = 0
    for i, cls in enumerate(classes):
        n_images = (
            len(list((source / cls).glob("*.[jJ][pP][gG]")))
            + len(list((source / cls).glob("*.[jJ][pP][eE][gG]")))
            + len(list((source / cls).glob("*.[pP][nN][gG]")))
        )
        print(f"   {i+1}. {cls}: {n_images} images")
        total_images += n_images

    temp_folder = Path("temp_dataset")
    if temp_folder.exists():
        shutil.rmtree(temp_folder)

    train_dir = temp_folder / "train"
    val_dir = temp_folder / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 Organisation du dataset (split {int((1-VALIDATION_SPLIT)*100)}/{int(VALIDATION_SPLIT*100)})...")

    for class_name in classes:
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        class_folder = source / class_name
        images = (
            list(class_folder.glob("*.[jJ][pP][gG]"))
            + list(class_folder.glob("*.[jJ][pP][eE][gG]"))
            + list(class_folder.glob("*.[pP][nN][gG]"))
        )

        train_imgs, val_imgs = train_test_split(images, test_size=VALIDATION_SPLIT, random_state=42)

        for img in train_imgs:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir / class_name / img.name)

        print(f"   ✓ {class_name}: {len(train_imgs)} train, {len(val_imgs)} validation")

    return train_dir, val_dir, classes


def create_dataloaders(train_dir, val_dir):
    # Data augmentation and normalization for training
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(40),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dataset = datasets.ImageFolder(train_dir, data_transforms["train"])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms["val"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader


def build_model(num_classes):
    print(f"\n🧠 Création du modèle MobileNetV2 ({num_classes} classes)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze the base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier
    # MobileNetV2 classifier is : Sequential(Dropout, Linear)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model.to(DEVICE)


class TrainingDashboard:
    def __init__(self, total_epochs=15, cols=5):
        self.total_epochs = total_epochs
        self.cols = cols
        self.rows = (total_epochs + cols - 1) // cols
        self.epoch_stats = [{"status": "WAIT"} for _ in range(total_epochs)]
        self.first_draw = True
        self.last_height = 0

    def update(self, epoch_idx, t_loss=None, t_acc=None, v_loss=None, v_acc=None, status="RUN"):
        if status == "RUN":
            self.epoch_stats[epoch_idx]["status"] = "RUN"
        elif status == "DONE":
            self.epoch_stats[epoch_idx] = {
                "status": "DONE",
                "t_loss": t_loss,
                "t_acc": t_acc,
                "v_loss": v_loss,
                "v_acc": v_acc,
            }
        else:
            self.epoch_stats[epoch_idx]["status"] = status

    def draw(self, pbar):
        # We construct the string to print
        lines = []
        for r in range(self.rows):
            header_str = ""
            line1_str = ""
            line2_str = ""
            for c in range(self.cols):
                idx = r * self.cols + c
                if idx < self.total_epochs:
                    stats = self.epoch_stats[idx]
                    status = stats["status"]
                    epo_label = f"Epoch {idx+1:02d}"
                    header_str += f"{epo_label:^14}   "

                    if status == "WAIT":
                        content_l1 = f"\033[90m{'WAIT':^14}\033[0m"
                        content_l2 = f"\033[90m{'. . .':^14}\033[0m"
                    elif status == "RUN":
                        content_l1 = f"\033[93m{'Running':^14}\033[0m"
                        content_l2 = f"\033[93m{'...':^14}\033[0m"
                    else:
                        # T:99% L.12
                        t_str = f"T:{stats['t_acc']*100:.0f}% L{stats['t_loss']:.2f}"
                        v_str = f"V:{stats['v_acc']*100:.0f}% L{stats['v_loss']:.2f}"
                        content_l1 = f"\033[92m{t_str:^14}\033[0m"
                        content_l2 = f"\033[96m{v_str:^14}\033[0m"

                    line1_str += f"[{content_l1}] "
                    line2_str += f"[{content_l2}] "

            lines.append("")  # Small spacer
            lines.append(header_str)
            lines.append(line1_str)
            lines.append(line2_str)
        lines.append("")

        # Determine movement height (how many lines we printed last time)
        # If it's not the first draw, we move up to overwrite
        if not self.first_draw:
            # Move up N lines
            print(f"\033[{self.last_height}A", end="")

        # Print lines with Clear Line code to avoid artifacts
        for line in lines:
            print(f"\033[2K{line}")  # Clear line then print

        self.last_height = len(lines)
        self.first_draw = False


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters of the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    patience_limit = 7  # Early stopping

    training_start_time = time.time()

    # Calculate total steps for the single global bar
    # We include validation steps so the bar moves during validation too
    total_steps = EPOCHS * (len(train_loader) + len(val_loader))

    # Initialize Dashboard
    dashboard = TrainingDashboard(EPOCHS)

    # Single global progress bar (positioned at bottom)
    # leave=True ensures it stays at the bottom
    pbar = tqdm(total=total_steps, unit="step", colour="green", leave=True)

    # Initial draw
    dashboard.draw(pbar)

    for epoch in range(EPOCHS):
        # Mark current epoch as running
        dashboard.update(epoch, status="RUN")
        dashboard.draw(pbar)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Update the global bar
                pbar.update(1)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # Store history
            if phase == "train":
                history["loss"].append(epoch_loss)
                history["accuracy"].append(epoch_acc.item())
            else:
                t_l = history["loss"][-1]
                t_a = history["accuracy"][-1]
                v_l = epoch_loss
                v_a = epoch_acc.item()
                history["val_loss"].append(v_l)
                history["val_accuracy"].append(v_a)

                dashboard.update(epoch, t_loss=t_l, t_acc=t_a, v_loss=v_l, v_acc=v_a, status="DONE")
                dashboard.draw(pbar)

                # Update scheduler
                scheduler.step(epoch_acc)

                # Deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0  # Reset patience
                else:
                    patience_counter += 1

        # Early stopping check
        if patience_counter >= patience_limit:
            pbar.write("🛑 Early stopping triggered!")
            break

    pbar.close()

    total_time = time.time() - training_start_time
    print(f"⏱️  Temps total: {format_time(total_time)}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history, best_acc, total_time


def plot_history(history, output_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Graphiques sauvegardés: {output_path}")


def save_package(model, classes, history, source_folder):
    output_name = Path(source_folder).name + "_model"
    output_folder = Path(output_name)
    output_folder.mkdir(exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), output_folder / "model.pth")
    print(f"💾 Modèle sauvegardé: {output_folder / 'model.pth'}")

    # Save metadata
    metadata = {
        "classes": classes,
        "img_size": IMG_SIZE,
        "num_classes": len(classes),
        "final_train_accuracy": float(history["accuracy"][-1]),
        "final_val_accuracy": float(history["val_accuracy"][-1]),
        "epochs": len(history["accuracy"]),
    }
    with open(output_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_history(history, output_folder / "training_history.png")

    shutil.make_archive(output_name, "zip", output_folder)
    print(f"📦 Package créé: {output_name}.zip")

    # SHA1 Signature
    sha1 = hashlib.sha1()
    with open(f"{output_name}.zip", "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)

    with open("signature.txt", "w") as f:
        f.write(f"{sha1.hexdigest()}  {output_name}.zip\n")

    shutil.rmtree(output_folder)
    return f"{output_name}.zip"


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <images_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    if not Path(source_folder).exists():
        print(f"❌ Erreur: {source_folder} n'existe pas")
        sys.exit(1)

    print("=" * 70)
    print("🌿 LEAFFLICTION")
    print("=" * 70)

    check_device()

    train_dir, val_dir, classes = organize_dataset(source_folder)

    print("\n🔧 Création des DataLoaders...")
    train_loader, val_loader = create_dataloaders(train_dir, val_dir)

    model = build_model(len(classes))

    model, history, final_acc, _ = train_model(model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"   Best Val Accuracy: {final_acc*100:.2f}%")

    if final_acc >= TARGET_ACCURACY:
        print(f"\n   ✅ Objectif atteint (>= {TARGET_ACCURACY*100}%)")
    else:
        print(f"\n   ⚠️  Objectif non atteint (< {TARGET_ACCURACY*100}%)")

    save_package(model, classes, history, source_folder)

    print("\n🧹 Nettoyage...")
    shutil.rmtree("temp_dataset")

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import timm


# =========================
# CONFIGURATION
# =========================

DATASET_NAME = "geraldmc/plantvillage-full"
DATASET_REVISION = "v0.1.0"

NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4

MODEL_PATH = "ml/checkpoints/best_efficientnet_b0.pth"


# =========================
# DEVICE
# =========================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

class PlantDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = item["class_idx"]

        image = self.transform(image)

        return image, label


# =========================
# LOAD DATA
# =========================

print("\nLoading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    revision=DATASET_REVISION,
    split="train"
)

train_data = dataset.filter(
    lambda x: x["split"] == "train"
)

test_data = dataset.filter(
    lambda x: x["split"] == "test"
)

print("Training images:", len(train_data))
print("Test images:", len(test_data))


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

indices = list(range(len(train_data)))

labels = train_data["class_idx"]

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.1,
    random_state=42,
    stratify=labels
)

train_subset = train_data.select(train_indices)
val_subset = train_data.select(val_indices)


print("Train:", len(train_subset))
print("Validation:", len(val_subset))


# =========================
# PYTORCH DATASETS
# =========================

train_dataset = PlantDataset(
    train_subset,
    train_transform
)

val_dataset = PlantDataset(
    val_subset,
    val_transform
)


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================

print("\nCreating EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================
# TRAINING
# =========================

best_val_accuracy = 0.0

os.makedirs(
    "ml/checkpoints",
    exist_ok=True
)


for epoch in range(EPOCHS):

    print(
        f"\n========== EPOCH {epoch + 1}/{EPOCHS} =========="
    )

    # ---------------------
    # TRAIN
    # ---------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        progress.set_postfix(
            loss=loss.item()
        )

    train_accuracy = (
        train_correct / train_total
    )

    # ---------------------
    # VALIDATION
    # ---------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc="Validation"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = (
        val_correct / val_total
    )

    print(
        f"\nTrain Loss: {train_loss / len(train_loader):.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss / len(val_loader):.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # ---------------------
    # SAVE BEST MODEL
    # ---------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1
            },
            MODEL_PATH
        )

        print(
            f"Saved best model → {MODEL_PATH}"
        )


print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print(
    f"Best validation accuracy: {best_val_accuracy:.4f}"
)

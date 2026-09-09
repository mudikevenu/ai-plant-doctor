import os
import torch
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np


DATASET_NAME = "geraldmc/plantvillage-full"
DATASET_REVISION = "v0.1.0"

MODEL_PATH = "ml/checkpoints/best_efficientnet_b0.pth"

NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 16


# =========================
# DEVICE
# =========================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
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

class PlantDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = item["class_idx"]

        image = transform(image)

        return image, label


# =========================
# LOAD DATA
# =========================

print("\nLoading test dataset...")

dataset = load_dataset(
    DATASET_NAME,
    revision=DATASET_REVISION,
    split="train"
)

test_data = dataset.filter(
    lambda x: x["split"] == "test"
)

print("Test images:", len(test_data))


test_dataset = PlantDataset(test_data)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================

print("\nLoading trained model...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================
# EVALUATION
# =========================

all_predictions = []
all_labels = []

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# =========================
# CLASS NAMES
# =========================

class_names = sorted(
    set(test_data["class_label"])
)


print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

os.makedirs(
    "ml/logs",
    exist_ok=True
)

np.save(
    "ml/logs/confusion_matrix.npy",
    cm
)

print(
    "\nConfusion matrix saved:"
    " ml/logs/confusion_matrix.npy"
)

print("\nEvaluation complete!")

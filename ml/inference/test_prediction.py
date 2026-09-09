from datasets import load_dataset
from PIL import Image
import os

print("Loading dataset...")

ds = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

# Find a test image
sample = next(
    x for x in ds
    if x["split"] == "test"
)

image = sample["image"]

os.makedirs("ml/datasets/processed", exist_ok=True)

path = "ml/datasets/processed/test_leaf.jpg"

image.save(path)

print("\nTest image saved:")
print(path)

print("\nActual label:")
print(sample["class_label"])

from datasets import load_dataset
from collections import Counter

print("Loading PlantVillage dataset...")

ds = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

print(f"\nTotal images: {len(ds):,}")

# Split distribution
split_counts = Counter(ds["split"])

print("\n===== SPLIT DISTRIBUTION =====")
for split, count in split_counts.items():
    print(f"{split}: {count:,}")

# Class distribution
class_counts = Counter(ds["class_label"])

print(f"\n===== NUMBER OF CLASSES =====")
print(len(class_counts))

print("\n===== CLASS DISTRIBUTION =====")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count:,}")

# Verify leaf grouping
leaf_grouped = Counter(ds["leaf_grouped"])

print("\n===== LEAF GROUPING =====")
for value, count in leaf_grouped.items():
    print(f"{value}: {count:,}")

# Display one sample
sample = ds[0]

print("\n===== SAMPLE =====")
print("Class:", sample["class_label"])
print("Disease:", sample["disease"])
print("Host:", sample["host"])
print("Image size:", sample["image"].size)
print("Leaf ID:", sample["leaf_id"])

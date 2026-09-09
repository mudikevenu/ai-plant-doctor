from datasets import load_dataset

print("Loading current PlantVillage Parquet dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    revision="refs/convert/parquet"
)

print("\n===== DATASET LOADED =====")
print(dataset)

for split in dataset:
    print(f"{split}: {len(dataset[split])} images")

print("\n===== FEATURES =====")
print(dataset["train"].features)

print("\n===== FIRST SAMPLE =====")
sample = dataset["train"][0]
print("Label:", sample["label"])
print("Crop:", sample["crop"])
print("Disease:", sample["disease"])
print("Leaf ID:", sample["leaf_id"])
print("Image:", sample["image"])

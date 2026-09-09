from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DATASET_NAME = "geraldmc/plantvillage-full"
DATASET_REVISION = "v0.1.0"


class PlantVillageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = item["image"].convert("RGB")
        label = item["class_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def load_plantvillage():
    print("Loading PlantVillage...")

    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="train"
    )

    train_data = dataset.filter(lambda x: x["split"] == "train")
    test_data = dataset.filter(lambda x: x["split"] == "test")

    print(f"Training images: {len(train_data):,}")
    print(f"Test images: {len(test_data):,}")

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_plantvillage()

    train_transform, eval_transform = get_transforms()

    train_dataset = PlantVillageDataset(
        train_data,
        train_transform
    )

    test_dataset = PlantVillageDataset(
        test_data,
        eval_transform
    )

    image, label = train_dataset[0]

    print("\n===== PIPELINE TEST =====")
    print("Image tensor shape:", image.shape)
    print("Label:", label)
    print("Image dtype:", image.dtype)

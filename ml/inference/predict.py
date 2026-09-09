import sys
import torch
import timm
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

MODEL_PATH = "ml/checkpoints/best_efficientnet_b0.pth"
NUM_CLASSES = 38
IMAGE_SIZE = 224

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("Device:", DEVICE)

# Load class names from dataset
dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

class_names = sorted(set(dataset["class_label"]))

# Load model
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_index = prediction.item()

    return (
        class_names[class_index],
        confidence.item()
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print(
            "python ml/inference/predict.py "
            "path/to/image.jpg"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    disease, confidence = predict(
        image_path
    )

    print("\n==============================")
    print("🌿 AI PLANT DOCTOR")
    print("==============================")

    print("Prediction:", disease)
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("==============================")

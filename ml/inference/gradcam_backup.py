import sys
import os
import torch
import timm
import numpy as np

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = "ml/checkpoints/best_efficientnet_b0.pth"
NUM_CLASSES = 38
IMAGE_SIZE = 224

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# Image
if len(sys.argv) != 2:
    print("Usage: python ml/inference/gradcam.py IMAGE_PATH")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit(1)

original = Image.open(image_path).convert("RGB")
original = original.resize((IMAGE_SIZE, IMAGE_SIZE))

rgb_image = np.array(original).astype(np.float32) / 255.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(original).unsqueeze(0).to(DEVICE)


# Prediction
with torch.no_grad():
    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

class_index = prediction.item()

print()
print("==============================")
print("🌿 AI PLANT DOCTOR")
print("==============================")
print("Prediction:", CLASS_NAMES[class_index])
print(f"Confidence: {confidence.item() * 100:.2f}%")


# Grad-CAM
target_layers = [model.conv_head]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [
    ClassifierOutputTarget(class_index)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# Save result
output_dir = "ml/logs/gradcam"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "gradcam_result.jpg"
)

Image.fromarray(
    visualization
).save(output_path)

print()
print("Grad-CAM saved:")
print(output_path)
print("==============================")

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    model,
    image,
    class_index,
    device,
    output_path
):
    """
    Generate a Grad-CAM visualization for an uploaded PIL image.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    rgb_image = np.array(image).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

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

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    Image.fromarray(
        visualization
    ).save(output_path)

    return output_path

import torch
import timm

NUM_CLASSES = 38


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_model(num_classes=NUM_CLASSES, pretrained=True):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


if __name__ == "__main__":
    device = get_device()

    print("Device:", device)

    model = create_model()
    model = model.to(device)

    dummy_input = torch.randn(
        2, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Number of classes:", NUM_CLASSES)
    print("Model test successful!")

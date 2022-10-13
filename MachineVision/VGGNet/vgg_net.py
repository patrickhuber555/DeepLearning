"""
The programme implements a CNN model inspired by the VGGNet. It achieves up to 93% accuracy.
"""
import os
from typing import Any

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from settings_vgg_net import (BATCH_SIZE, EPOCHS, FILENAME, NUM_CLASSES,
                              PATH_SAVE_MODEL, TRAIN_DATA_PATH, VAL_DATA_PATH)


class VGGNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # first Conv-Pool-Block
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Batch Normalisation
            nn.BatchNorm2d(128),
            # second Conv-Pool-Block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Batch Normalisation
            nn.BatchNorm2d(256),
            # third Conv-Pool Block
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Batch Normalisation
            nn.BatchNorm2d(512),
            # fourth Conv-Pool-Block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            # fifth Conv-Pool-Block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 17),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(17, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def check_image(path: str) -> bool:
    """
    Auxiliary function. Checks whether the images can be processed without problems.

    Args:
        path: Path to the images.

    Returns:
        True, if everything is OK. False if there are issues.
    """
    try:
        im = Image.open(path)
        return True
    except:
        return False


def load_data() -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Creates the training and validation dataset and creates the dataloader for it.

    Returns:
        tuple[DataLoader[Any], DataLoader[Any]]: Validation and test data set.
    """
    img_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = ImageFolder(
        root=TRAIN_DATA_PATH, transform=img_transforms, is_valid_file=check_image
    )

    val_data = ImageFolder(
        root=VAL_DATA_PATH, transform=img_transforms, is_valid_file=check_image
    )

    # create dataloader ojects
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=True
    )
    return train_data_loader, val_data_loader


def get_device() -> torch.device:
    """
    Looks to see if the GPU is available. If the GPU is not available, the CPU is used.

    Returns:
        GPU or CPU.
    """
    if torch.cuda.is_available():
        print("GPU is activated.")
        device = torch.device("cuda")
    else:
        print("CPU is activated.")
        device = torch.device("cpu")
    return device


def train(
    model: VGGNet,
    optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_loader,
    val_loader,
    epochs: int = 20,
    device: torch.device = "cpu",
):
    """
    This is a complete training and validation loop in which all components (loss function and optimiser) can be passed
    as parameters.

    Args:
        model: The model is based on the VGG-Net.
        optimizer: Adam-based optimiser.
        loss_fn: CrossEntropyLoss for multi-category classification.
        train_loader: Iterable object with the dataset of the training data.
        val_loader: Iterable object with the dataset of the validation data.
        epochs: Number of passes with which the data passes through the network (default value: 20).
        device: CPU or GPU to process the tensors (default value: cpu).

    Returns:
        None.
    """
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            f"Epoch: {epoch}, Training Loss: {training_loss:.2f}, Validation Loss: {valid_loss:.2f}, accuracy = {num_correct / num_examples:.2F}"
        )


def save_model(net: VGGNet) -> None:
    """
    Speichert das Model/Netzwerk ab.

    Args:
        net: Das Netzwerk/Model, das gespeichert wird.

    Returns:
        None.
    """
    torch.save(net, os.path.join(PATH_SAVE_MODEL, FILENAME))


def main():
    print("Я очень люблю тебя, Наталья")
    train_data_loader, val_data_loader = load_data()
    device = get_device()
    vgg_net = VGGNet(num_classes=NUM_CLASSES)
    print(vgg_net)
    optimizer = optim.Adam(vgg_net.parameters(), lr=0.001)
    vgg_net.to(device)
    train(
        vgg_net,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_data_loader,
        val_data_loader,
        epochs=EPOCHS,
        device=device,
    )
    save_model(vgg_net)


if __name__ == "__main__":
    main()

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os

from settings_fashion_mnist_LeNet5 import (
    DOWNLOAD_DIR,
    BATCH_SIZE,
    EPOCHS,
    N_CLASSES,
    FILENAME,
    PATH_SAVE_MODEL,
)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 600),  # Totales Fragezeichen !!!!!!!
            nn.Dropout(p=0.5),
            # nn.Linear(600, N_CLASSES),
            nn.Linear(600, 120),
            nn.Dropout(p=0.5),
            nn.Linear(120, N_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    # Download und Laden des Trainingsdatensatzes(60k)
    train_data = datasets.FashionMNIST(
        DOWNLOAD_DIR, download=True, train=True, transform=transform
    )
    val_data = datasets.FashionMNIST(
        DOWNLOAD_DIR, download=True, train=False, transform=transform
    )
    # Dataloader-Objekte erstellen
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_data_loader, val_data_loader


def get_device():
    if torch.cuda.is_available():
        print("GPU ist aktiviert.")
        device = torch.device("cuda")
    else:
        print("CPU ist aktiviert.")
        device = torch.device("cpu")
    return device


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
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


def save_model(net):
    torch.save(net, os.path.join(PATH_SAVE_MODEL, FILENAME))


def main():
    print("Я очень люблю тебя, Наталья")
    train_data_loader, val_data_loader = load_data()
    le_net = LeNet()
    print(le_net)
    device = get_device()
    optimizer = optim.Adam(le_net.parameters(), lr=0.001)
    le_net.to(device)
    train(
        le_net,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_data_loader,
        val_data_loader,
        epochs=EPOCHS,
        device=device,
    )
    save_model(le_net)


if __name__ == "__main__":
    main()

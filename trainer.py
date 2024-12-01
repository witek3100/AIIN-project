import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision

from model import Model

NUM_EPOCHS = 1


class Trainer:
    """
    Class responsible for model initialisation and training for given solution.

    **kwargs contains training and model parameters
    """
    def __init__(self, datasets_path: str = './data', **kwargs):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),

            torchvision.transforms.RandomCrop(kwargs.get('conv_neurons')[0], padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=datasets_path, train=True, download=True, transform=transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=datasets_path, train=False, download=True, transform=transforms)

        batch_size = kwargs.get('batch_size')

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Model(**kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get("learning_rate", 1e-4))

        self.criterion = torch.nn.CrossEntropyLoss()

    def run(self):
        """
        Performs model training, returns accuracy on test dataset afterwards.

        :return:  accuracy on test dataset
        """
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(self.train_loader):.4f}")

        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            final_accuracy = 100 * correct / total

        return final_accuracy

import torch
import torch.nn as nn
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import os

#os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin" #necessary line to run on local machine 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_set = Flowers102(root='data', split='test', download=True, transform=test_transform)

test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class FlowerClassifier(nn.Module):
    def __init__(self):
        super(FlowerClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 102)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model
model = FlowerClassifier().to(device)

example_input, labels = next(iter(test_loader))
example_input = example_input.to(device)
output = model(example_input)


figure = make_dot(output, params=dict(model.named_parameters()))

figure.format = "png"
figure.attr(dpi="300") #Clearer Image
figure.render("NN_GRAPH")

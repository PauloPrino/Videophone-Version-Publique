import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

num_classes = 5

# Définition du modèle CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Transformations pour les images d'entrée
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger le modèle sauvegardé
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('person_recognition_1.pth'))
model.eval()

# Fonction pour reconnaître une personne
def recognize_person(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        # Remplacez 'classes' par la liste réelle des noms de classes
        classes = ['Hugo', 'Isabelle', 'Marc', 'Marie', 'Paul']  # Exemple de liste de classes
        return classes[predicted.item()]

files = os.listdir("Labeled_data/Data/Hugo")

for image in files:
    image_path = os.path.join("Labeled_data/Data/Hugo", image)
    predicted_class = recognize_person(image_path, model)
    print(f'Predicted class for image {image}: {predicted_class}')

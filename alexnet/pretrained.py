import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import alexnet

from PIL import Image

alexnet = alexnet(pretrained=True)

alexnet.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),       # Resize to 256 pixels
    transforms.CenterCrop(224),   # Crop to 224x224 (required for AlexNet)
    transforms.ToTensor(),        # Convert the image to a PyTorch tensor
    transforms.Normalize(         # Normalize using ImageNet's mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

img_path = './example.jpeg'
img = Image.open(img_path)

img_tensor = preprocess(img)

img_tensor = img_tensor.unsqueeze(0)

with torch.no_grad():
    outputs = alexnet(img_tensor)

print(outputs, outputs.size())

data, predicted_class = torch.max(outputs, 1)

print(f'Predicted class index: {predicted_class.item()}')

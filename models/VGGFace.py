import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.optim import Adam
from torch import nn


class



# Initialize the MTCNN module for face detection (optional)
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load your dataset
dataset = YourDataset('path_to_your_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the InceptionResnetV1 model for fine-tuning
model = InceptionResnetV1(pretrained='vggface2').to('cuda' if torch.cuda.is_available() else 'cpu')

# Modify the final fully connected layer to output 2 classes (you, not you)
model.classify = True
model.logits = nn.Linear(model.last_linear.in_features, 2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

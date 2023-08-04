import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# # Define the transformations: resize and convert to tensor
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor()
# ])

# # Load the data
# data_dir = '/home/hades/salima/image_processing_tool/images'
# dataset = ImageFolder(data_dir, transform=transform)
# data_loader = DataLoader(dataset, batch_size=128, shuffle=True)


import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Define the directory containing the data
data_dir = '/home/hades/salima/image_processing_tool/images'

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image pixel values
])

# Load the dataset
dataset = ImageFolder(data_dir, transform=transform)

# Create a data loader to iterate over the dataset in batches
batch_size = 128
shuffle = True
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 64 * 64 * 3), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Train the autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    for img, _ in data_loader:
        img = img.view(img.size(0), -1)
        img = img.cuda()
        # Forward
        output = model(img)
        loss = criterion(output, img)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print out the loss periodically.
    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 10

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])




train_dataset = '/home/hades/salima/image_processing_tool/images'
data_dir = train_dataset
dataset = ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the autoencoder model
model = Autoencoder().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch_images, _ in train_loader:
        batch_images = batch_images.view(-1, 784).to(device)
        
        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'autoencoder.pth')

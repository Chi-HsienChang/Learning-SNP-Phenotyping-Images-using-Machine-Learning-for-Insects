import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

def make_image_transparent(img_path):
    img = Image.open(img_path).convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        # change all white (also shades of whites)
        # pixels to transparent
        if item[0] in list(range(200, 256)):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image = make_image_transparent(image_path)  # Make the image transparent
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, _):
        image = self.image
        if self.transform is not None:
            image = self.transform(image)
        return image

# Define the transformation to apply to the images
image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = SingleImageDataset('./wing_image_for_ML_gray/AR0507.jpg', transform=image_transform)

# Create the DataLoader
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # Notice the number of channels (4) to accommodate for RGBA images
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(images)
        loss = loss_function(recon_batch, images)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        original_images = images.view(images.shape[0], 4, 200, 200).cpu().detach()
        recon_images = recon_batch.view(recon_batch.shape[0], 4, 200, 200).cpu().detach()

        # Save the original and reconstructed images for every epoch
        if batch_idx == 0:  # Save the first batch of each epoch
            # Save the original images
            for i, image in enumerate(original_images):
                image_tensor = image.squeeze()  # remove dimensions of size 1
                image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
                pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
                pil_image = pil_image.convert('RGBA')  # Convert to grayscale
                pil_image.save('./original_images/epoch_{}_batch_{}_image_{}.png'.format(epoch, batch_idx, i))
            
            # Save the reconstructed images
            for i, image in enumerate(recon_images):
                image_tensor = image.squeeze()  # remove dimensions of size 1
                image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
                pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
                pil_image = pil_image.convert('RGBA')  # Convert to grayscale
                pil_image.save('./recon_images/epoch_{}_batch_{}_image_{}.png'.format(epoch, batch_idx, i))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


epochs = 2000
for epoch in range(1, epochs + 1):
    train(epoch)




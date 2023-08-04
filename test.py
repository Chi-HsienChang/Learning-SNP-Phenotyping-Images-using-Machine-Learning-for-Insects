import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class WingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform is not None:
            image = self.transform(image)
        return image

# Define the transformation to apply to the images
image_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = WingDataset('./wing_image_resized', transform=image_transform)

# Create the DataLoader
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AE(nn.Module):
    def __init__(self, dims):
        super(AE, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, dims),
            nn.Sigmoid()
        )


        self.decoder_lin = nn.Sequential(
            nn.Linear(dims, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.decoder_unflatten = nn.Unflatten(1, unflattened_size=(128, 8, 8))

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        x1 = self.encoder_cnn(x)
        x1 = self.flatten(x1)
        x1 = nn.Dropout(0.3)(x1)
        x1 = self.encoder_lin(x1)
        
        x2 = self.decoder_lin(x1)
        x2 = self.decoder_unflatten(x2)
        x2 = self.decoder_cnn(x2)
        return x2, x1

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    best_loss = float('inf')
    best_image = None
    best_original = None
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(images)
        loss = loss_function(recon_batch, images)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_image = recon_batch.view(recon_batch.shape[0], 1, 300, 300).cpu().detach()
            best_original = images.view(images.shape[0], 1, 300, 300).cpu().detach()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return best_image, best_original



latent_dims = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(dims=latent_dims).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 2000
for epoch in range(1, epochs + 1):
    best_image, best_original = train(epoch)

    # Save the best image
    if best_image is not None and best_original is not None:
        # Save the best reconstruction
        image_tensor = best_image[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image = pil_image.convert('L')  # Convert to grayscale
        pil_image.save('AE_image/best_image_epoch_{}.png'.format(epoch))

        # Save the corresponding original image
        image_tensor = best_original[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image = pil_image.convert('L')  # Convert to grayscale
        pil_image.save('AE_image/best_original_epoch_{}.png'.format(epoch))


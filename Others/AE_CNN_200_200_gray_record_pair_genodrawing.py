import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pdb 

class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.latent_space = dims
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
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
        # pdb.set_trace()
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

    def forward(self, x1):
        x1 = x1.view(8192, -1) # salima
        x1 = self.encoder_cnn(x1)
        x1 = self.flatten(x1)
        x1 = nn.Dropout(0.3)(x1)
        x = self.encoder_lin(x1)
        return x

class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.latent_space = dims
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
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
            nn.Sigmoid(),

        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.decoder_unflatten(x)
        x = self.decoder_cnn(x)

        return x

class WingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform is not None:
            image = self.transform(image)
        return image

# Define the transformation to apply to the images
image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = WingDataset('./wing_image_for_ML_gray', transform=image_transform)

# Create the DataLoader
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded



def loss_function(recon_x, x):
    BCE = F.mse_loss(recon_x, x, reduction='sum')  # Use Mean Square Error for RGB images
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
            best_image = recon_batch.view(recon_batch.shape[0], 1, 200, 200).cpu().detach()
            best_original = images.view(images.shape[0], 1, 200, 200).cpu().detach()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return best_image, best_original


input_dim = 200 * 200
hidden_dim = 400
latent_dim = 100
output_dim = 200 * 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(latent_dim=latent_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 20
for epoch in range(1, epochs + 1):
    best_image, best_original = train(epoch)

    # Save the best image
    if best_image is not None and best_original is not None:
        # Save the best reconstruction
        image_tensor = best_image[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image = pil_image.convert('L')  # Convert to grayscale
        pil_image.save('best_image_epoch_{}.png'.format(epoch))

        # Save the corresponding original image
        image_tensor = best_original[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image = pil_image.convert('L')  # Convert to grayscale
        pil_image.save('best_original_epoch_{}.png'.format(epoch))


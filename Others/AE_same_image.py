import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
class SingleImageDataset():
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, _):
        image = Image.open(self.image_path).convert('L')  # Convert to grayscale
        if self.transform is not None:
            image = self.transform(image)
        return image

# Define the transformation to apply to the images
image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])



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
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# # Create the dataset
# dataset = WingDataset('./wing_image_for_ML_gray', transform=image_transform)


# Create the dataset
dataset = SingleImageDataset('./wing_image_for_ML_gray/AR0507.jpg', transform=image_transform)

# Create the DataLoader
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# Create the DataLoader
batch_size = 128
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE


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

        # Save the original and reconstructed images for every epoch
        if batch_idx == 0:  # Save the first batch of each epoch
            original_images = images.view(images.shape[0], 1, 200, 200).cpu().detach()
            recon_images = recon_batch.view(recon_batch.shape[0], 1, 200, 200).cpu().detach()

            # Save the original images
            for i, image in enumerate(original_images):
                image_tensor = image.squeeze()  # remove dimensions of size 1
                image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
                pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
                pil_image = pil_image.convert('L')  # Convert to grayscale
                pil_image.save('./original_images/epoch_{}_batch_{}_image_{}.png'.format(epoch, batch_idx, i))
            
            # Save the reconstructed images
            for i, image in enumerate(recon_images):
                image_tensor = image.squeeze()  # remove dimensions of size 1
                image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
                pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
                pil_image = pil_image.convert('L')  # Convert to grayscale
                pil_image.save('./recon_images/epoch_{}_batch_{}_image_{}.png'.format(epoch, batch_idx, i))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))





input_dim = 200 * 200
hidden_dim = 400
latent_dim = 100
output_dim = 200 * 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


epochs = 2000
for epoch in range(1, epochs + 1):
    train(epoch)
# epochs = 2000
# for epoch in range(1, epochs + 1):
#     image, original = train(epoch)

#     # Save the image
#     image_tensor = image[0].squeeze()  # remove dimensions of size 1
#     image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
#     pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
#     pil_image = pil_image.convert('L')  # Convert to grayscale
#     pil_image.save('image_epoch_{}.png'.format(epoch))

#     # Save the corresponding original image
#     image_tensor = original[0].squeeze()  # remove dimensions of size 1
#     image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale pixel intensities to [0, 255]
#     pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
#     pil_image = pil_image.convert('L')  # Convert to grayscale
#     pil_image.save('original_epoch_{}.png'.format(epoch))
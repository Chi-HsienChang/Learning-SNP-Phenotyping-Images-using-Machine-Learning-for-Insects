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
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset
dataset = WingDataset('./wing_image_for_ML_B_and_W', transform=image_transform)

# Create the DataLoader
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        z = self.fc2(h1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, input_dim))
        recon_batch = self.decode(z)
        return recon_batch, z


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, output_dim), reduction='sum')
    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    best_loss = float('inf')
    best_image = None
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(images)
        loss = loss_function(recon_batch, images.view(-1, output_dim))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_image = recon_batch.view(recon_batch.shape[0], 1, 28, 28).cpu().detach()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return best_image


input_dim = 784  # 28 * 28
hidden_dim = 400
latent_dim = 20
output_dim = 784  # 28 * 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(1, epochs + 1):
    best_image = train(epoch)

    # Save the best image
    if best_image is not None:
        image_tensor = best_image[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 0.5 + 0.5) * 255  # Scale back the pixel intensities
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image.save('best_image_epoch_{}.png'.format(epoch))

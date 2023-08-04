import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class SNPsDataset(Dataset):
    def __init__(self, snps, mnist):
        self.snps = snps
        self.mnist = mnist

    def __len__(self):
        return len(self.snps)

    def __getitem__(self, idx):
        return self.snps[idx], self.mnist[idx]


# Generate random SNP data and load MNIST data
snps = np.random.choice([0, 1], size=(60000, 500))  # MNIST has 60k images
mnist = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
mnist = mnist.data.float() / 255.0  # Normalize to [0,1]

# Convert numpy array to PyTorch tensors and create a dataset
snps_torch = torch.from_numpy(snps).float()
dataset = SNPsDataset(snps_torch, mnist)

# Create a DataLoader
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
    for batch_idx, (snps, images) in enumerate(train_loader):
        snps = snps.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        recon_batch, _ = model(snps)
        loss = loss_function(recon_batch, images.view(-1, output_dim))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_image = recon_batch.view(recon_batch.shape[0], 1, 28, 28).cpu().detach()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return best_image


input_dim = 500
hidden_dim = 400
latent_dim = 20
output_dim = 28 * 28  # MNIST images are 28x28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(1, epochs + 1):
    best_image = train(epoch)

    # Save the best image
    if best_image is not None:
        image_tensor = best_image[0].squeeze()  # take the first image of the batch and remove dimensions of size 1
        image_tensor = (image_tensor * 255).type(torch.uint8)  # Scale back the pixel intensities
        pil_image = transforms.ToPILImage()(image_tensor)  # Convert to PIL image
        pil_image.save('best_image_epoch_{}.png'.format(epoch))

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SNPsDataset(Dataset):
    def __init__(self, snps):
        self.snps = snps

    def __len__(self):
        return len(self.snps)

    def __getitem__(self, idx):
        return self.snps[idx]

# Let's generate a sample data of 1000 individuals each having 500 SNPs
num_individuals = 1000
num_snps = 500

# Generate random SNP data
snps = np.random.choice([0, 1], size=(num_individuals, num_snps))

# Convert numpy array to PyTorch tensors and create a dataset
snps_torch = torch.from_numpy(snps).float()
dataset = SNPsDataset(snps_torch)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvariance layer
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))  # Adjusted here
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')  # Adjusted here
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):  # Adjusted here (removed unused _)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# ... (Your DataLoader and SNP generation code here) ...

input_dim = 500
hidden_dim = 400
latent_dim = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch)

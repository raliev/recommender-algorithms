# app/algorithms/autoencoder.py
import numpy as np
from .base import Recommender

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class VAEModel(nn.Module):
    def __init__(self, num_items, latent_dim=64, hidden_dim=128):
        super(VAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2) # mu and log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

class VAERecommender(Recommender):
    def __init__(self, k, epochs=20, batch_size=64, learning_rate=0.001, **kwargs):
        super().__init__(k) # k is the latent dimension
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch' to use this recommender.")

        self.name = "VAE"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.num_items = None
        self.reconstructed_matrix = None

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        # Using multinomial likelihood for implicit feedback
        log_softmax = torch.log_softmax(recon_x, dim=1)
        recon_loss = -torch.sum(log_softmax * x)

        # KL Divergence
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld

    def fit(self, R, progress_callback=None, visualizer = None):
        _, self.num_items = R.shape

        # Binarize the input for implicit feedback
        R_binary = (R > 0).astype(np.float32)

        dataset = TensorDataset(torch.FloatTensor(R_binary))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = VAEModel(num_items=self.num_items, latent_dim=self.k)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data in dataloader:
                inputs = data[0]
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(inputs)
                loss = self.loss_function(recon_batch, inputs, mu, log_var)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            if progress_callback:
                progress_callback((epoch + 1) / self.epochs)
        return self

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            # To predict, we pass the binarized training data through the model
            # to get the reconstructed "probability" distribution for each user.
            R_binary = (self.train_data > 0).astype(np.float32)
            inputs = torch.FloatTensor(R_binary)
            self.R_predicted, _, _ = self.model(inputs)

            # For visualization, let's also store the reconstructed matrix
            self.reconstructed_matrix = self.R_predicted.numpy()

        return self.R_predicted.numpy()
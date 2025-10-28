import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import urllib.request
import zipfile

# Function to download and extract MovieLens 100k dataset if not present
def download_movielens():
    if not os.path.exists('ml-100k'):
        print("Downloading MovieLens 100k dataset...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        urllib.request.urlretrieve(url, 'ml-100k.zip')
        with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
            zip_ref.extractall()
        os.remove('ml-100k.zip')
        print("Dataset downloaded and extracted.")
    else:
        print("MovieLens 100k dataset already exists.")

# PMF Model Class
class PMF:
    def __init__(self, num_users, num_items, latent_dim=2, lambda_u=0.1, lambda_v=0.1, lr=0.01, num_epochs=100):
        # Initialize latent factors with Gaussian priors (mean 0, small variance for regularization effect)
        self.U = np.random.normal(0, 0.1, (num_users, latent_dim))  # User latents
        self.V = np.random.normal(0, 0.1, (num_items, latent_dim))  # Item latents
        self.lambda_u = lambda_u  # Regularization for users (from prior variance)
        self.lambda_v = lambda_v  # Regularization for items (from prior variance)
        self.lr = lr  # Learning rate
        self.num_epochs = num_epochs
        self.latent_dim = latent_dim

    def compute_prediction(self, u, i):
        return np.dot(self.U[u], self.V[i])

    def compute_rmse(self, ratings):
        # Compute RMSE over given ratings (list of (u, i, r))
        errors = []
        for u, i, r in ratings:
            pred = self.compute_prediction(u, i)
            errors.append((r - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def compute_loss(self, ratings):
        # Compute regularized squared loss (equivalent to negative log-posterior in PMF)
        loss = 0
        reg_u = 0
        reg_v = 0
        for u, i, r in ratings:
            pred = self.compute_prediction(u, i)
            loss += (r - pred) ** 2 / 2  # From Gaussian likelihood
        reg_u = self.lambda_u / 2 * np.sum(np.linalg.norm(self.U, axis=1) ** 2)  # From Gaussian prior on U
        reg_v = self.lambda_v / 2 * np.sum(np.linalg.norm(self.V, axis=1) ** 2)  # From Gaussian prior on V
        return loss + reg_u + reg_v

    def train(self, train_ratings, val_ratings, plot_interval=10, save_dir='pmf_visuals'):
        # Create directory for saving plots
        os.makedirs(save_dir, exist_ok=True)

        train_losses = []
        val_losses = []
        train_rmses = []
        val_rmses = []

        for epoch in range(self.num_epochs):
            # Shuffle training ratings for SGD
            np.random.shuffle(train_ratings)

            # SGD updates
            for u, i, r in train_ratings:
                pred = self.compute_prediction(u, i)
                err = r - pred
                # Update user latent
                self.U[u] += self.lr * (err * self.V[i] - self.lambda_u * self.U[u])
                # Update item latent
                self.V[i] += self.lr * (err * self.U[u] - self.lambda_v * self.V[i])

            # Compute metrics
            train_loss = self.compute_loss(train_ratings)
            val_loss = self.compute_loss(val_ratings)
            train_rmse = self.compute_rmse(train_ratings)
            val_rmse = self.compute_rmse(val_ratings)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_rmses.append(train_rmse)
            val_rmses.append(val_rmse)

            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

            # Plot and save visuals at intervals
            if (epoch + 1) % plot_interval == 0 or epoch == 0:
                self.plot_loss(train_losses, val_losses, epoch, save_dir)
                self.plot_rmse(train_rmses, val_rmses, epoch, save_dir)
                self.plot_latent_space(epoch, save_dir)
                self.plot_latent_histograms(epoch, save_dir)
                self.plot_sample_predictions(train_ratings[:10], epoch, save_dir)  # Sample of 10 for demo

    def plot_loss(self, train_losses, val_losses, epoch, save_dir):
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Regularized Loss Over Epochs (Up to {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'loss_epoch_{epoch+1}.png'))
        plt.close()  # Close to avoid overlapping in memory

    def plot_rmse(self, train_rmses, val_rmses, epoch, save_dir):
        plt.figure(figsize=(8, 6))
        plt.plot(train_rmses, label='Train RMSE')
        plt.plot(val_rmses, label='Val RMSE')
        plt.title(f'RMSE Over Epochs (Up to {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'rmse_epoch_{epoch+1}.png'))
        plt.close()

    def plot_latent_space(self, epoch, save_dir):
        if self.latent_dim != 2:
            print("Latent space visualization only for dim=2.")
            return
        plt.figure(figsize=(8, 6))
        plt.scatter(self.U[:, 0], self.U[:, 1], label='Users', alpha=0.5)
        plt.scatter(self.V[:, 0], self.V[:, 1], label='Items', alpha=0.5)
        plt.title(f'2D Latent Space at Epoch {epoch+1}')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'latent_space_epoch_{epoch+1}.png'))
        plt.close()

    def plot_latent_histograms(self, epoch, save_dir):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.U.flatten(), bins=50, alpha=0.7)
        plt.title(f'User Latents Histogram at Epoch {epoch+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(self.V.flatten(), bins=50, alpha=0.7)
        plt.title(f'Item Latents Histogram at Epoch {epoch+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'latent_hist_epoch_{epoch+1}.png'))
        plt.close()

    def plot_sample_predictions(self, sample_ratings, epoch, save_dir):
        actuals = [r for _, _, r in sample_ratings]
        preds = [self.compute_prediction(u, i) for u, i, _ in sample_ratings]
        indices = range(len(actuals))

        plt.figure(figsize=(8, 6))
        plt.bar(indices, actuals, width=0.4, label='Actual Ratings', alpha=0.7)
        plt.bar([i + 0.4 for i in indices], preds, width=0.4, label='Predicted Ratings', alpha=0.7)
        plt.title(f'Sample Predictions vs Actuals at Epoch {epoch+1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Rating')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'sample_preds_epoch_{epoch+1}.png'))
        plt.close()

# Main script
if __name__ == "__main__":
    download_movielens()

    # Load ratings data (user_id, item_id, rating, timestamp)
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Adjust IDs to 0-index
    ratings_df['user_id'] -= 1
    ratings_df['item_id'] -= 1

    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['item_id'].nunique()

    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # Convert to list of tuples (u, i, r)
    ratings = list(zip(ratings_df['user_id'], ratings_df['item_id'], ratings_df['rating']))

    # Split into train and validation (80/20)
    train_ratings, val_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    # Initialize and train PMF model (latent_dim=2 for 2D visualization)
    pmf = PMF(num_users, num_items, latent_dim=2, lambda_u=0.1, lambda_v=0.1, lr=0.001, num_epochs=100)
    pmf.train(train_ratings, val_ratings, plot_interval=10)

    print("Training complete. Visualizations saved in 'pmf_visuals' directory.")
    # Note: You can screenshot the saved PNG files for your article.
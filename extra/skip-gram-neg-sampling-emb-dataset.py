import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import requests
import zipfile
import io

# --- Step 1: Download and Load Data ---
# This part handles downloading the MovieLens 100K dataset automatically.
def download_movielens_100k():
    """Downloads and extracts the MovieLens 100K dataset."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    print("Downloading MovieLens 100K dataset...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Extracting files...")
    z.extractall()
    print("Dataset downloaded and extracted to 'ml-100k' directory.")

try:
    # Try loading data assuming it's already downloaded
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]
except FileNotFoundError:
    # If not found, download it
    download_movielens_100k()
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]

# --- Step 2: Prepare "Sentences" and Vocabulary ---
# A "sentence" is a list of movies a user has rated highly (e.g., rating >= 4)
print("Preparing data...")
# Filter for high ratings
high_ratings_df = ratings_df[ratings_df['rating'] >= 4]

# Group movies by user to create our "sentences"
sentences = high_ratings_df.groupby('user_id')['movie_id'].apply(list).tolist()

# Build vocabulary of unique movies
all_movies = list(ratings_df['movie_id'].unique())
vocab_size = len(all_movies)
movie_to_idx = {movie: i for i, movie in enumerate(all_movies)}
idx_to_movie = {i: movie for movie, i in movie_to_idx.items()}

# --- Step 3: Compute Negative Sampling Distribution ---
# Same logic: frequency of each movie raised to the power of 0.75
freq = defaultdict(int)
for sentence in sentences:
    for movie_id in sentence:
        freq[movie_id] += 1

freq_list = [freq[movie_id] for movie_id in all_movies]
prob = np.array(freq_list) ** 0.75
prob /= prob.sum()
prob_tensor = torch.from_numpy(prob).float()

# --- Step 4: Generate Positive Co-occurrence Pairs ---
positive_pairs = []
for sentence in sentences:
    if len(sentence) < 2:
        continue
    # Convert movie_ids to vocabulary indices
    indices = [movie_to_idx[movie_id] for movie_id in sentence]
    for i in range(len(indices)):
        for j in range(len(indices)):
            if i != j:
                positive_pairs.append((indices[i], indices[j]))
print(f"Generated {len(positive_pairs)} positive pairs.")

# --- Step 5: Create Dataset and DataLoader ---
class PairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = torch.tensor(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

dataset = PairsDataset(positive_pairs)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# --- Step 6: Define the Skip-gram with Negative Sampling Model ---
# This model is identical to your provided code.
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        # Initialize weights
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward(self, centers, contexts, negatives):
        v_c = self.in_embed(centers)      # batch_size x embed_dim
        v_o = self.out_embed(contexts)    # batch_size x embed_dim
        pos_dot = (v_c * v_o).sum(dim=1)  # batch_size
        pos_loss = torch.log(torch.sigmoid(pos_dot))

        v_neg = self.out_embed(negatives) # batch_size x k x embed_dim
        # Reshape v_c for batch matrix multiplication
        v_c_unsqueezed = v_c.unsqueeze(2) # batch_size x embed_dim x 1
        neg_dot = torch.bmm(v_neg, v_c_unsqueezed).squeeze(2) # batch_size x k
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(dim=1)

        loss = - (pos_loss + neg_loss).mean()
        return loss

# --- Step 7: Train the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embed_dim = 50
k = 5  # Number of negative samples
epochs = 10
lr = 0.001

model = SkipGramNS(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Starting training...")
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        centers, contexts = batch[:, 0].to(device), batch[:, 1].to(device)

        # Generate negative samples for the current batch
        negs = torch.multinomial(
            prob_tensor, len(centers) * k, replacement=True
        ).view(len(centers), k).to(device)

        loss = model(centers, contexts, negs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

# --- Step 8: Extract Embeddings and Make Recommendations ---
# Get the learned input embeddings
embeddings = model.in_embed.weight.data.cpu().numpy()

# Create a mapping from movie_id to title for display
id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))

def get_similar_movies(target_movie_title, top_n=10):
    """Finds movies similar to a target movie using cosine similarity."""
    try:
        # Find the movie_id for the given title
        target_movie_id = movies_df[movies_df['title'] == target_movie_title]['movie_id'].iloc[0]

        # Get the index and embedding vector for the target movie
        target_idx = movie_to_idx[target_movie_id]
        target_vec = embeddings[target_idx]

        # Calculate cosine similarities
        similarities = (embeddings @ target_vec) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec)
        )

        # Get the indices of the most similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        print(f"\nMovies similar to '{target_movie_title}':")
        for idx in similar_indices:
            movie_id = idx_to_movie[idx]
            sim_score = similarities[idx]
            print(f"- {id_to_title.get(movie_id, 'Unknown')}: Similarity = {sim_score:.4f}")

    except IndexError:
        print(f"Movie with title '{target_movie_title}' not found in the dataset.")

# --- Step 9: Get Recommendations for a few movies ---
get_similar_movies("Star Wars (1977)")
get_similar_movies("Toy Story (1995)")
get_similar_movies("Scream (1996)")
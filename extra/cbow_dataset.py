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
import itertools # Needed for CBOW context generation

# --- Step 1: Download and Load Data ---
# (No changes needed here - keep the original code)
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
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]
except FileNotFoundError:
    download_movielens_100k()
    ratings_df = pd.read_csv(
        'ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies_df = pd.read_csv(
        'ml-100k/u.item', sep='|', encoding='latin-1', header=None,
        names=['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    )[['movie_id', 'title']]

# --- Step 2: Prepare "Sentences" and Vocabulary ---
# (No changes needed here - keep the original code)
print("Preparing data...")
high_ratings_df = ratings_df[ratings_df['rating'] >= 4]
sentences = high_ratings_df.groupby('user_id')['movie_id'].apply(list).tolist()
all_movies = sorted(list(ratings_df['movie_id'].unique())) # Use sorted list for consistent mapping
vocab_size = len(all_movies)
movie_to_idx = {movie: i for i, movie in enumerate(all_movies)}
idx_to_movie = {i: movie for movie, i in movie_to_idx.items()}

# --- Step 3: Compute Negative Sampling Distribution ---
# (No changes needed here - keep the original code)
freq = defaultdict(int)
for sentence in sentences:
    for movie_id in sentence:
        if movie_id in movie_to_idx: # Ensure movie is in vocabulary
            freq[movie_to_idx[movie_id]] += 1 # Use index for frequency count

# Ensure freq_list aligns with the vocabulary indices (0 to vocab_size-1)
freq_list = [freq[i] for i in range(vocab_size)]
prob = np.array(freq_list) ** 0.75
prob += 1e-8 # Add small epsilon to avoid zero probability if a movie has 0 freq
prob /= prob.sum()
prob_tensor = torch.from_numpy(prob).float()

# --- Step 4: Generate CBOW Training Examples ---
# *** MODIFIED FOR CBOW ***
# Create (context_indices, center_index) pairs
cbow_examples = []
max_context_size = 0 # Keep track for potential padding later if needed
for sentence in sentences:
    if len(sentence) < 2:
        continue
    # Convert movie_ids to vocabulary indices
    indices = [movie_to_idx[movie_id] for movie_id in sentence if movie_id in movie_to_idx]
    if len(indices) < 2:
        continue

    # Create examples: (context=[all other indices], center=current index)
    for i in range(len(indices)):
        center_index = indices[i]
        context_indices = indices[:i] + indices[i+1:]
        if not context_indices: # Skip if context is empty
            continue
        cbow_examples.append((context_indices, center_index))
        if len(context_indices) > max_context_size:
            max_context_size = len(context_indices)

print(f"Generated {len(cbow_examples)} CBOW examples.")
# print(f"Max context size: {max_context_size}") # Optional: Check max context length

# --- Step 5: Create Dataset and DataLoader ---
# *** MODIFIED FOR CBOW ***
class CBOWDataset(Dataset):
    def __init__(self, examples):
        # Store context and center separately for easier batching if needed later
        self.contexts = [torch.tensor(ctx, dtype=torch.long) for ctx, _ in examples]
        self.centers = torch.tensor([ctr for _, ctr in examples], dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        # Return context tensor and center index
        return self.contexts[idx], self.centers[idx]

# Custom collate function to handle variable length contexts by padding
def pad_collate(batch):
    # Separate contexts and centers
    contexts = [item[0] for item in batch]
    centers = torch.stack([item[1].unsqueeze(0) for item in batch]).squeeze() # Stack centers into a tensor

    # Pad contexts
    # nn.EmbeddingBag handles variable lengths directly if mode='mean', no padding needed then.
    # If using regular nn.Embedding and manual averaging, padding *would* be needed.
    # Let's assume we'll use EmbeddingBag for simplicity as it handles averaging.
    # So, we just need the contexts as a list of tensors and the centers as a tensor.
    # However, for negative sampling, we need consistent shapes. Let's use EmbeddingBag approach.

    # We need offsets for EmbeddingBag if batching variable lengths
    offsets = [0] + [len(ctx) for ctx in contexts]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    contexts_cat = torch.cat(contexts)

    return contexts_cat, offsets, centers


dataset = CBOWDataset(cbow_examples)
# Use the custom collate function
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=pad_collate)


# --- Step 6: Define the CBOW with Negative Sampling Model ---
# *** MODIFIED FOR CBOW ***
class CBOWNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # EmbeddingBag efficiently computes sums/means of embeddings for variable lengths
        # mode='mean' directly gives the averaged context vector
        self.in_embed = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.out_embed = nn.Embedding(vocab_size, embed_dim) # Predict center word embedding

        # Initialize weights (optional but good practice)
        self.in_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.out_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)


    def forward(self, contexts, offsets, centers, negatives):
        # contexts: concatenated indices of all contexts in the batch
        # offsets: starting index of each context sequence in 'contexts'
        # centers: target center word indices for each context (batch_size)
        # negatives: negative sample indices (batch_size x k)

        # Get averaged context vectors using EmbeddingBag
        avg_context_vectors = self.in_embed(contexts, offsets) # batch_size x embed_dim

        # Get target embeddings (positive samples)
        v_o = self.out_embed(centers) # batch_size x embed_dim

        # Calculate positive dot products
        pos_dot = (avg_context_vectors * v_o).sum(dim=1) # batch_size
        pos_loss = torch.log(torch.sigmoid(pos_dot))

        # Get negative sample embeddings
        v_neg = self.out_embed(negatives) # batch_size x k x embed_dim

        # Calculate negative dot products (using averaged context vector)
        # Reshape avg_context_vectors for batch matrix multiplication
        avg_context_unsqueezed = avg_context_vectors.unsqueeze(2) # batch_size x embed_dim x 1
        neg_dot = torch.bmm(v_neg, avg_context_unsqueezed).squeeze(2) # batch_size x k
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(dim=1) # batch_size

        loss = - (pos_loss + neg_loss).mean()
        return loss

# --- Step 7: Train the Model ---
# *** MODIFIED FOR CBOW ***
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embed_dim = 50
k = 5  # Number of negative samples
epochs = 10
lr = 0.005 # CBOW might benefit from slightly higher LR initially

# Instantiate the CBOW model
model = CBOWNS(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Starting CBOW training...")
for epoch in range(epochs):
    total_loss = 0
    for contexts_cat, offsets, centers in dataloader:
        contexts_cat = contexts_cat.to(device)
        offsets = offsets.to(device)
        centers = centers.to(device)

        # Generate negative samples for the current batch
        # Number of samples needed = batch_size * k
        num_centers_in_batch = centers.size(0)
        negs = torch.multinomial(
            prob_tensor, num_centers_in_batch * k, replacement=True
        ).view(num_centers_in_batch, k).to(device)

        # Ensure negative samples are not the actual center word (optional but good practice)
        # This part can be complex and slow, often skipped for performance.
        # Simple check (might not be perfectly efficient):
        # for batch_idx in range(num_centers_in_batch):
        #    center_val = centers[batch_idx].item()
        #    while center_val in negs[batch_idx]:
        #        new_negs = torch.multinomial(prob_tensor, k, replacement=True).to(device)
        #        negs[batch_idx] = new_negs

        # Pass data to the CBOW model
        loss = model(contexts_cat, offsets, centers, negs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

# --- Step 8: Extract Embeddings and Make Recommendations ---
# (No changes needed here - use in_embed weights from CBOW)
print("Training finished. Extracting embeddings...")
# Use the input embeddings learned by EmbeddingBag
embeddings = model.in_embed.weight.data.cpu().numpy()

# Create a mapping from movie_id to title for display
id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))

def get_similar_movies(target_movie_title, top_n=10):
    """Finds movies similar to a target movie using cosine similarity."""
    try:
        target_movie_id = movies_df[movies_df['title'] == target_movie_title]['movie_id'].iloc[0]
        if target_movie_id not in movie_to_idx:
            print(f"Movie ID {target_movie_id} for title '{target_movie_title}' not found in training vocabulary.")
            return

        target_idx = movie_to_idx[target_movie_id]
        target_vec = embeddings[target_idx]

        similarities = (embeddings @ target_vec) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec)
        )
        # Add epsilon to prevent division by zero if norm is zero
        similarities = np.nan_to_num(similarities) # Replace NaN with 0

        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        print(f"\nMovies similar to '{target_movie_title}':")
        for idx in similar_indices:
            # Check if index is valid before accessing idx_to_movie
            if idx < len(all_movies):
                movie_id = idx_to_movie[idx]
                sim_score = similarities[idx]
                print(f"- {id_to_title.get(movie_id, f'Unknown ID: {movie_id}')}: Similarity = {sim_score:.4f}")
            else:
                print(f"Warning: Invalid index {idx} encountered during similarity search.")


    except IndexError:
        print(f"Movie with title '{target_movie_title}' not found in the movie metadata.")
    except KeyError:
        print(f"Movie ID corresponding to '{target_movie_title}' not found in mapping.")

# --- Step 9: Get Recommendations for a few movies ---
# (No changes needed here)
get_similar_movies("Star Wars (1977)")
get_similar_movies("Toy Story (1995)")
get_similar_movies("Scream (1996)")
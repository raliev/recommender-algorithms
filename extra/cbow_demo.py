import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import itertools # Needed for CBOW context generation

# Set random seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)

# --- Step 1: Create a simple, illustrative dataset ---
# *** MODIFIED TO CREATE MORE DATA ***
# The previous version failed because 11 training examples is not enough.
# We'll create 50 "users" for each profile to give the model
# enough data to learn the co-occurrence patterns.

print("Creating simple illustrative dataset...")

# Define our movie metadata (the "vocabulary")
movies_data = [
    [1, 'Star Battles: Part 1'], [2, 'Star Battles: Part 2'], [3, 'Star Battles: Part 3'],
    [4, 'Lord of the Rings: Part 1'], [5, 'Lord of the Rings: Part 2'], [6, 'Lord of the Rings: Part 3'],
    [7, 'Funny Movie 1'], [8, 'Funny Movie 2'],
    [9, 'Just Action'],
    [10, 'Sci-Fi 4']
]
movies_df = pd.DataFrame(movies_data, columns=['movie_id', 'title'])

# Define the user "profiles" or "sentences"
profile_scifi = [1, 2, 3, 10]
profile_fantasy = [4, 5, 6]
profile_comedy = [7, 8]
profile_bridge = [1, 4]  # The user linking Sci-Fi and Fantasy
profile_isolate = [9]    # The user who likes one "unpopular" movie

ratings_data = []
user_counter = 1

# Create 50 users for each profile
num_users_per_profile = 50

for i in range(num_users_per_profile):
    # Sci-Fi Fans
    for movie_id in profile_scifi:
        ratings_data.append([f'user_{user_counter}', movie_id, 5, 0])
    user_counter += 1

    # Fantasy Fans
    for movie_id in profile_fantasy:
        ratings_data.append([f'user_{user_counter}', movie_id, 5, 0])
    user_counter += 1

    # Comedy Fans
    for movie_id in profile_comedy:
        ratings_data.append([f'user_{user_counter}', movie_id, 5, 0])
    user_counter += 1

    # Bridge Fans
    for movie_id in profile_bridge:
        ratings_data.append([f'user_{user_counter}', movie_id, 5, 0])
    user_counter += 1

# Add one "isolate" user
for movie_id in profile_isolate:
    ratings_data.append([f'user_{user_counter}', movie_id, 5, 0])
user_counter += 1


ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating', 'timestamp'])

print(f"Dataset created with {user_counter - 1} total users.")
# print(ratings_df.head(10)) # Uncomment to check


# --- Step 2: Prepare "Sentences" and Vocabulary ---
# (No changes needed, this logic is dataset-agnostic)
print("\nPreparing data...")
high_ratings_df = ratings_df[ratings_df['rating'] >= 4]
sentences = high_ratings_df.groupby('user_id')['movie_id'].apply(list).tolist()
all_movies = sorted(list(ratings_df['movie_id'].unique())) # Use sorted list for consistent mapping
vocab_size = len(all_movies)
movie_to_idx = {movie: i for i, movie in enumerate(all_movies)}
idx_to_movie = {i: movie for movie, i in movie_to_idx.items()}

print(f"\nVocabulary Size: {vocab_size}")
# print(f"Movie to Index map: {movie_to_idx}") # Uncomment to debug
# print("Generated 'sentences' (movies liked by each user):") # Uncomment to debug
# for s in sentences[:5]: # Print first 5
#     print(s)


# --- Step 3: Compute Negative Sampling Distribution ---
# (No changes needed, this logic is dataset-agnostic)
freq = defaultdict(int)
for sentence in sentences:
    for movie_id in sentence:
        if movie_id in movie_to_idx: # Ensure movie is in vocabulary
            freq[movie_to_idx[movie_id]] += 1 # Use index for frequency count

freq_list = [freq[i] for i in range(vocab_size)]
prob = np.array(freq_list) ** 0.75
prob += 1e-8 # Add small epsilon
prob /= prob.sum()
prob_tensor = torch.from_numpy(prob).float()

# --- Step 4: Generate CBOW Training Examples ---
# (No changes needed, this logic is dataset-agnostic)
cbow_examples = []
max_context_size = 0
for sentence in sentences:
    if len(sentence) < 2:
        continue
    indices = [movie_to_idx[movie_id] for movie_id in sentence if movie_id in movie_to_idx]
    if len(indices) < 2:
        continue

    for i in range(len(indices)):
        center_index = indices[i]
        context_indices = indices[:i] + indices[i+1:]
        if not context_indices: # Skip if context is empty
            continue
        cbow_examples.append((context_indices, center_index))
        if len(context_indices) > max_context_size:
            max_context_size = len(context_indices)

# This number should now be much larger (50 * (4+3+2+2)) = 550
print(f"\nGenerated {len(cbow_examples)} CBOW examples.")


# --- Step 5: Create Dataset and DataLoader ---
class CBOWDataset(Dataset):
    def __init__(self, examples):
        self.contexts = [torch.tensor(ctx, dtype=torch.long) for ctx, _ in examples]
        self.centers = torch.tensor([ctr for _, ctr in examples], dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.contexts[idx], self.centers[idx]

def pad_collate(batch):
    contexts = [item[0] for item in batch]
    centers = torch.stack([item[1].unsqueeze(0) for item in batch]).squeeze()
    offsets = [0] + [len(ctx) for ctx in contexts]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    contexts_cat = torch.cat(contexts)
    return contexts_cat, offsets, centers


dataset = CBOWDataset(cbow_examples)
# *** MODIFIED BATCH SIZE ***
# Batch size 4 is too small for 550 examples. Let's use 16.
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)


# --- Step 6: Define the CBOW with Negative Sampling Model ---
# (No changes needed)
class CBOWNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        self.in_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.out_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, contexts, offsets, centers, negatives):
        avg_context_vectors = self.in_embed(contexts, offsets)
        v_o = self.out_embed(centers)
        pos_dot = (avg_context_vectors * v_o).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_dot))

        v_neg = self.out_embed(negatives)
        avg_context_unsqueezed = avg_context_vectors.unsqueeze(2)
        neg_dot = torch.bmm(v_neg, avg_context_unsqueezed).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(dim=1)

        loss = - (pos_loss + neg_loss).mean()
        return loss

# --- Step 7: Train the Model ---
# *** MODIFIED HYPERPARAMETERS FOR CONVERGENCE ***
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

embed_dim = 10
k = 5  # Number of negative samples

# *** THE FIX: Lower LR, more Epochs ***
epochs = 2000
lr = 0.001 # Was 0.01, which was too high and prevented convergence

model = CBOWNS(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Starting CBOW training...")
for epoch in range(epochs):
    total_loss = 0
    for contexts_cat, offsets, centers in dataloader:
        contexts_cat = contexts_cat.to(device)
        offsets = offsets.to(device)
        centers = centers.to(device)

        num_centers_in_batch = centers.size(0)
        negs = torch.multinomial(
            prob_tensor, num_centers_in_batch * k, replacement=True
        ).view(num_centers_in_batch, k).to(device)

        loss = model(contexts_cat, offsets, centers, negs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # *** MODIFIED PRINT FREQUENCY ***
    # Print loss every 200 epochs
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

# --- Step 8: Extract Embeddings and Make Recommendations ---
# (No changes needed, but I removed the special "zero-vector" check
# as it's not necessary. An untrained vector will just have random similarity.)
print("Training finished. Extracting embeddings...")
embeddings = model.in_embed.weight.data.cpu().numpy()
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

        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            target_norm = 1e-10 # Avoid division by zero

        embedding_norms = np.linalg.norm(embeddings, axis=1)
        embedding_norms[embedding_norms == 0] = 1e-10

        similarities = (embeddings @ target_vec) / (embedding_norms * target_norm)
        similarities = np.nan_to_num(similarities) # Replace NaN with 0

        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]

        print(f"\nMovies similar to '{target_movie_title}':")
        for idx in similar_indices:
            if idx < len(all_movies):
                movie_id = idx_to_movie[idx]
                sim_score = similarities[idx]
                # Filter out low/random similarities
                if sim_score > 0.1:
                    print(f"- {id_to_title.get(movie_id, f'Unknown ID: {movie_id}')}: Similarity = {sim_score:.4f}")

    except IndexError:
        print(f"Movie with title '{target_movie_title}' not found in the movie metadata.")
    except KeyError:
        print(f"Movie ID corresponding to '{target_movie_title}' not found in mapping.")

# --- Step 9: Get Recommendations for a few movies ---
# (No changes needed)

# Test Case 1: Sci-Fi.
# Should be similar to other Sci-Fi AND the Fantasy group (due to User 4)
get_similar_movies("Star Battles: Part 1")

# Test Case 2: Fantasy.
# Should be similar to other Fantasy AND the Sci-Fi group (due to User 4)
get_similar_movies("Lord of the Rings: Part 2")

# Test Case 3: Comedy.
# Should ONLY be similar to other Comedy.
get_similar_movies("Funny Movie 1")

# Test Case 4: Isolate.
# Was never in a training context. Its vector is untrained.
# Should have no meaningful similarities.
get_similar_movies("Just Action")
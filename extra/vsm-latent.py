import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os

# --- Model Definition ---
class SkipGramNS(nn.Module):
    """
    Skip-gram model with Negative Sampling.
    It learns embeddings for a vocabulary of items (in this case, genres).
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        # Initialize weights for better training performance
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward(self, centers, contexts, negatives):
        # Forward pass implementing the loss function from the Word2Vec paper
        v_c = self.in_embed(centers)
        v_o = self.out_embed(contexts)

        # Positive score: dot product of center and context embeddings
        pos_dot = (v_c * v_o).sum(1)
        pos_loss = torch.log(torch.sigmoid(pos_dot))

        # Negative score: dot product of center and negative sample embeddings
        v_neg = self.out_embed(negatives)
        # Unsqueeze for batch matrix multiplication
        v_c_unsq = v_c.unsqueeze(2)
        neg_dot = torch.bmm(v_neg, v_c_unsq).squeeze(2)
        # We want the sigmoid of the negative dot product to be close to 1
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(1)

        # The total loss is the negative sum of positive and negative sample losses
        loss = - (pos_loss + neg_loss).mean()
        return loss

# --- Recommendation Function ---
def get_similar_movies(target_movie_title, movie_df, movie_genres, genre_to_idx, genre_embeddings, movie_ids_list, movie_matrix, top_n=5):
    """
    Finds and prints the top N most similar movies to a target movie
    based on the cosine similarity of their aggregated genre vectors.
    """
    # Find the movie_id for the target title
    target_id_row = movie_df[movie_df['title'] == target_movie_title]
    if target_id_row.empty:
        print(f"Movie '{target_movie_title}' not found.")
        return

    target_id = target_id_row['movie_id'].iloc[0]

    # Check if the movie is in our processed list (it might have had no genres)
    if target_id not in movie_ids_list:
        print(f"Movie '{target_movie_title}' was not processed (it may have no genres).")
        return

    target_idx_in_list = movie_ids_list.index(target_id)

    # Get the vector for the target movie
    target_vec = movie_matrix[target_idx_in_list]

    # Calculate cosine similarity between the target vector and all other movie vectors
    # Formula: (A . B) / (||A|| * ||B||)
    sims = (movie_matrix @ target_vec) / (np.linalg.norm(movie_matrix, axis=1) * np.linalg.norm(target_vec))

    # Get the indices of the top N most similar movies (the first one is the movie itself)
    similar_indices = np.argsort(sims)[::-1][1:top_n+1]

    print(f"Movies similar to '{target_movie_title}' (Genres: {movie_genres[target_id]}):")
    for idx in similar_indices:
        sim_movie_id = movie_ids_list[idx]
        sim_movie_title = movie_df[movie_df['movie_id'] == sim_movie_id]['title'].iloc[0]
        similarity_score = sims[idx]
        print(f"- {sim_movie_title} (Genres: {movie_genres[sim_movie_id]}): Sim = {similarity_score:.4f}")

# --- Main Execution Block ---
if __name__ == "__main__":

    # Check for dataset
    data_path = 'ml-100k/u.item'
    if not os.path.exists(data_path):
        print("Error: The 'ml-100k' directory was not found.")
        print("Please download the MovieLens 100K dataset and place it in the same directory as this script.")
        exit()

    # --- Step 1: Prepare Data from Movie Genres ---
    print("--- Step 1: Preparing Data ---")

    movie_df = pd.read_csv(data_path, sep='|', encoding='latin-1', header=None,
                           names=['movie_id', 'title'] + list(range(22)))

    genre_cols = movie_df.columns[-19:]
    movie_genres = {}
    for index, row in movie_df.iterrows():
        genres = [genre_cols[i] for i, val in enumerate(row[genre_cols]) if val == 1]
        if genres:
            movie_genres[row['movie_id']] = genres

    all_genres = sorted(list(set(g for genres in movie_genres.values() for g in genres)))
    genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
    vocab_size = len(all_genres)
    print(f"Found {len(movie_genres)} movies with genres and {vocab_size} unique genres.")

    positive_pairs = []
    for genres in movie_genres.values():
        if len(genres) >= 2:
            indices = [genre_to_idx[g] for g in genres]
            for i in range(len(indices)):
                for j in range(len(indices)):
                    if i != j:
                        positive_pairs.append((indices[i], indices[j]))
    print(f"Generated {len(positive_pairs)} positive genre pairs.")

    genre_freq = defaultdict(int)
    for genres in movie_genres.values():
        for g in genres:
            genre_freq[g] += 1

    freq_list = [genre_freq[g] for g in all_genres]
    prob = np.array(freq_list) ** 0.75
    prob /= prob.sum()
    prob_tensor = torch.from_numpy(prob).float()

    class GenrePairsDataset(Dataset):
        def __len__(self):
            return len(positive_pairs)
        def __getitem__(self, idx):
            return torch.tensor(positive_pairs[idx])

    dataloader = DataLoader(GenrePairsDataset(), batch_size=1024, shuffle=True)

    # --- Step 2: Train the Model to Learn Genre Embeddings ---
    print("\n--- Step 2: Training Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SkipGramNS(vocab_size, embed_dim=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    k, epochs = 5, 20

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            centers, contexts = batch[:, 0].to(device), batch[:, 1].to(device)
            negs = torch.multinomial(prob_tensor, len(centers) * k, replacement=True)
            negs = negs.view(len(centers), k).to(device)

            loss = model(centers, contexts, negs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Avg Loss = {total_loss / len(dataloader):.4f}")

    genre_embeddings = model.in_embed.weight.data.cpu().numpy()

    # --- Step 3: Generate Content-Based Recommendations ---
    print("\n--- Step 3: Generating Recommendations ---")

    movie_vectors = {}
    for movie_id, genres in movie_genres.items():
        if genres:
            genre_indices = [genre_to_idx[g] for g in genres]
            movie_vectors[movie_id] = genre_embeddings[genre_indices].mean(axis=0)

    movie_ids_list = list(movie_vectors.keys())
    movie_matrix = np.array([movie_vectors[mid] for mid in movie_ids_list])

    # Get recommendations for a few example movies
    get_similar_movies(
        'Toy Story (1995)',
        movie_df=movie_df,
        movie_genres=movie_genres,
        genre_to_idx=genre_to_idx,
        genre_embeddings=genre_embeddings,
        movie_ids_list=movie_ids_list,
        movie_matrix=movie_matrix,
        top_n=5
    )

    print("\n")

    get_similar_movies(
        'Star Wars (1977)',
        movie_df=movie_df,
        movie_genres=movie_genres,
        genre_to_idx=genre_to_idx,
        genre_embeddings=genre_embeddings,
        movie_ids_list=movie_ids_list,
        movie_matrix=movie_matrix,
        top_n=5
    )

    print("\n")

    get_similar_movies(
        'Scream (1996)',
        movie_df=movie_df,
        movie_genres=movie_genres,
        genre_to_idx=genre_to_idx,
        genre_embeddings=genre_embeddings,
        movie_ids_list=movie_ids_list,
        movie_matrix=movie_matrix,
        top_n=5
    )
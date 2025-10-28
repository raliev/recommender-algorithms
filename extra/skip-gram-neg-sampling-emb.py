import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from io import StringIO
from torch.utils.data import Dataset, DataLoader

# Real dataset: Top 20 highest-grossing films (from Wikipedia, public for research)
csv_data = """movie_id,title,year,genres,gross
1,Avatar,2009,Action|Adventure|Fantasy,2923710708
2,Avengers: Endgame,2019,Action|Adventure|Sci-Fi,2797501328
3,Avatar: The Way of Water,2022,Action|Adventure|Fantasy,2320250281
4,Titanic,1997,Drama|Romance,2187463944
5,Star Wars: The Force Awakens,2015,Action|Adventure|Sci-Fi,2068223624
6,Avengers: Infinity War,2018,Action|Adventure|Sci-Fi,2048359754
7,Spider-Man: No Way Home,2021,Action|Adventure|Sci-Fi,1922598800
8,Jurassic World,2015,Action|Adventure|Sci-Fi,1671713208
9,The Lion King,2019,Animation|Adventure|Drama,1656943394
10,The Avengers,2012,Action|Adventure|Sci-Fi,1518812988
11,Furious 7,2015,Action|Crime|Thriller,1515047671
12,Avengers: Age of Ultron,2015,Action|Adventure|Sci-Fi,1405403694
13,Black Panther,2018,Action|Adventure|Sci-Fi,1346739107
14,Harri Potter and the Deathly Hallows – Part 2,2011,Adventure|Family|Fantasy,1342139277
15,Star Wars: The Last Jedi,2017,Action|Adventure|Sci-Fi,1332539971
16,Jurassic World: Fallen Kingdom,2018,Action|Adventure|Sci-Fi,1303459585
17,Frozen II,2019,Animation|Adventure|Comedy,1450026933
18,Beauty and the Beast,2017,Family|Fantasy|Musical,1263521126
19,Incredibles 2,2018,Animation|Action|Adventure,1242805359
20,The Fate of the Furious,2017,Action|Crime|Thriller,1234846438"""

df = pd.read_csv(StringIO(csv_data))
movies = {row['title']: row['genres'].split('|') for _, row in df.iterrows()}

# Step 1: Build vocabulary from unique features
all_features = set()
for feats in movies.values():
    all_features.update(feats)
features = list(all_features)
vocab_size = len(features)
feature_to_idx = {f: i for i, f in enumerate(features)}

# Step 2: Compute feature frequencies for negative sampling distribution (unigram^0.75)
freq = {f: 0 for f in features}
for feats in movies.values():
    for f in feats:
        freq[f] += 1
freq_list = [freq[f] for f in features]
prob = np.array(freq_list) ** 0.75
prob /= prob.sum()
prob_tensor = torch.from_numpy(prob).float()

# Step 3: Generate positive pairs (center, context) from co-occurring features in each movie
positive_pairs = []
for feats in movies.values():
    if len(feats) < 2:
        continue
    feat_indices = [feature_to_idx[f] for f in feats]
    for i in range(len(feat_indices)):
        for j in range(len(feat_indices)):
            if i != j:
                positive_pairs.append((feat_indices[i], feat_indices[j]))

# Step 4: Dataset for positive pairs
class PairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = torch.tensor(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

dataset = PairsDataset(positive_pairs)

# Step 5: Skip-gram with Negative Sampling model
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, centers, contexts, negatives):
        v_c = self.in_embed(centers)  # batch_size x embed_dim
        v_o = self.out_embed(contexts)  # batch_size x embed_dim
        pos_dot = (v_c * v_o).sum(dim=1)  # batch_size
        pos_loss = torch.log(torch.sigmoid(pos_dot))  # batch_size

        v_neg = self.out_embed(negatives)  # batch_size x k x embed_dim
        neg_dot = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)  # batch_size x k
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(dim=1)  # batch_size

        loss = - (pos_loss + neg_loss).mean()
        return loss

# Step 6: Train the model
embed_dim = 5  # Small dimension for toy example
model = SkipGramNS(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
k = 5  # Number of negative samples per positive

for epoch in range(200):
    total_loss = 0
    for batch in dataloader:
        centers, contexts = batch[:, 0], batch[:, 1]
        negs = torch.multinomial(prob_tensor, len(centers) * k, replacement=True).view(len(centers), k)
        loss = model(centers, contexts, negs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

# Step 7: Extract feature embeddings (use input embeddings)
embeddings = model.in_embed.weight.data.cpu().numpy()

# Step 8: Compute item vectors by averaging feature embeddings
item_vectors = {}
for name, feats in movies.items():
    if feats:
        vecs = [embeddings[feature_to_idx[f]] for f in feats]
        item_vectors[name] = np.mean(vecs, axis=0)

# Step 9: Recommendation - Compute cosine similarities to a target item
target = 'Avatar'
print(f"\nSimilarities to {target}:")
for other in movies:
    if other != target:
        sim = np.dot(item_vectors[target], item_vectors[other]) / (
                np.linalg.norm(item_vectors[target]) * np.linalg.norm(item_vectors[other])
        )
        print(f"{other}: {sim:.4f}")
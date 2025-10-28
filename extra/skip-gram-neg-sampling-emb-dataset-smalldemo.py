import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from io import StringIO
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

plot_dir = "embedding_plots"
os.makedirs(plot_dir, exist_ok=True)
print(f"Графики будут сохранены в папку: {plot_dir}")
# ---

# Исходный датасет: Top 20 фильмов
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

all_features = set()
for feats in movies.values():
    all_features.update(feats)
features = list(all_features)
vocab_size = len(features)
feature_to_idx = {f: i for i, f in enumerate(features)}
idx_to_feature = {i: f for f, i in feature_to_idx.items()} # Для меток на графике

freq = {f: 0 for f in features}
for feats in movies.values():
    for f in feats:
        freq[f] += 1
freq_list = [freq[f] for f in features]
prob = np.array(freq_list) ** 0.75
prob /= prob.sum()
prob_tensor = torch.from_numpy(prob).float()

positive_pairs = []
for feats in movies.values():
    if len(feats) < 2:
        continue
    feat_indices = [feature_to_idx[f] for f in feats]
    for i in range(len(feat_indices)):
        for j in range(len(feat_indices)):
            if i != j:
                positive_pairs.append((feat_indices[i], feat_indices[j]))

class PairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = torch.tensor(pairs)
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

dataset = PairsDataset(positive_pairs)

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, centers, contexts, negatives):
        v_c = self.in_embed(centers)
        v_o = self.out_embed(contexts)
        pos_dot = (v_c * v_o).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_dot))
        v_neg = self.out_embed(negatives)
        neg_dot = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_dot)).sum(dim=1)
        loss = - (pos_loss + neg_loss).mean()
        return loss

def plot_embeddings(embeddings, epoch_num, save_path):
    plt.figure(figsize=(10, 8))
    for i in range(len(embeddings)):
        x, y = embeddings[i]
        plt.scatter(x, y, s=100, alpha=0.7)
        plt.text(x + 0.03, y + 0.03, idx_to_feature[i], fontsize=9) # Добавляем метку
    
    plt.title(f'Жанровые Эмбеддинги (2D) - Эпоха {epoch_num}', fontsize=14)
    plt.xlabel('Измерение 1', fontsize=12)
    plt.ylabel('Измерение 2', fontsize=12)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig(save_path)
    plt.close()

embed_dim = 2
model = SkipGramNS(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
k = 5
num_epochs = 150 # Установим 150 эпох для наглядности

# --- Новое: Списки для хранения истории ---
epoch_losses = []
plot_intervals = [1, 25, 50, 100, 150] # Эпохи, в которые мы сохраняем "снимки"
# ---

print("Начало обучения...")
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        centers, contexts = batch[:, 0], batch[:, 1]
        negs = torch.multinomial(prob_tensor, len(centers) * k, replacement=True).view(len(centers), k)
        loss = model(centers, contexts, negs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss) # Сохраняем loss

    # --- Новое: Логика сохранения "снимков" ---
    if (epoch + 1) in plot_intervals:
        print(f"Эпоха {epoch + 1}/{num_epochs}, Средний Loss: {avg_loss:.4f}")
        current_embeds = model.in_embed.weight.data.cpu().numpy().copy()
        plot_path = os.path.join(plot_dir, f'embeddings_epoch_{epoch+1}.png')
        plot_embeddings(current_embeds, epoch + 1, plot_path)
        print(f"   -> Снимок эмбеддингов сохранен в {plot_path}")
    # ---

print("Обучение завершено.")

# --- Новое: Отрисовка графика потерь ---
loss_plot_path = os.path.join(plot_dir, 'loss_vs_epoch.png')
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title('Loss Модели vs. Эпоха', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Средний Loss', fontsize=12)
plt.grid(True)
plt.savefig(loss_plot_path)
plt.close()
print(f"График Loss сохранен в {loss_plot_path}")
# ---

# Шаги 7-9: Используем эмбеддинги для рекомендации (без изменений)
# (Это полезно, чтобы увидеть, как 2D-векторы все еще могут давать осмысленные результаты)

# Шаг 7: Извлечение эмбеддингов
embeddings = model.in_embed.weight.data.cpu().numpy()

# Шаг 8: Вычисление векторов фильмов
item_vectors = {}
for name, feats in movies.items():
    if feats:
        vecs = [embeddings[feature_to_idx[f]] for f in feats]
        item_vectors[name] = np.mean(vecs, axis=0)

# Шаг 9: Рекомендации
target = 'Avatar'
print(f"\nСходство с фильмом '{target}' (на основе 2D векторов):")
sim_scores = []
for other in movies:
    if other != target:
        if target in item_vectors and other in item_vectors:
            sim = np.dot(item_vectors[target], item_vectors[other]) / (
                    np.linalg.norm(item_vectors[target]) * np.linalg.norm(item_vectors[other])
            )
            sim_scores.append((other, sim))

# Сортировка по убыванию сходства
sim_scores.sort(key=lambda x: x[1], reverse=True)
for other, sim in sim_scores[:10]: # Показываем топ-10
    print(f"{other}: {sim:.4f}")
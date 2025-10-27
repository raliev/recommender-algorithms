# app/algorithms/sasrec.py
import numpy as np
from .base import Recommender

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class SASRecModel(nn.Module):
    def __init__(self, num_items, k, num_blocks, num_heads, dropout_rate, max_len):
        super(SASRecModel, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 1, k, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, k)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=k, nhead=num_heads, dim_feedforward=k * 4, dropout=dropout_rate, batch_first=True),
            num_layers=num_blocks
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq):
        # seq shape: (batch_size, seq_len)
        seq_len = seq.size(1)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)

        # Embeddings and positional encoding
        x = self.item_embedding(seq) + self.positional_embedding(positions)
        x = self.dropout(x)

        # Causal mask to prevent attending to future items
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=seq.device)

        # Self-attention encoder
        log_feats = self.encoder(x, mask=mask)

        return log_feats

    def predict(self, seq):
        log_feats = self.forward(seq) # (batch, seq_len, k)
        final_feat = log_feats[:, -1, :] # Use the output of the last item in the sequence

        # Dot product with all item embeddings to get scores
        item_embeds = self.item_embedding.weight
        logits = torch.matmul(final_feat, item_embeds.transpose(0, 1))
        return logits


class SeqDataset(Dataset):
    def __init__(self, R, max_len):
        self.users = range(R.shape[0])
        self.user_items = {u: R[u, :].nonzero()[0] for u in self.users}
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_items[user]

        # Pad or truncate sequence
        padded_seq = np.zeros(self.max_len, dtype=np.int64)
        if len(seq) > 0:
            end = min(len(seq), self.max_len)
            padded_seq[-end:] = seq[-end:]

        # The input is the sequence, the target is the next item
        input_seq = padded_seq[:-1]
        target_item = padded_seq[1:]

        return torch.LongTensor(input_seq), torch.LongTensor(target_item)


class SASRecRecommender(Recommender):
    def __init__(self, k, epochs=30, batch_size=128, learning_rate=0.001, max_len=50, num_blocks=2, num_heads=1, dropout_rate=0.2, **kwargs):
        super().__init__(k)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch' to use this recommender.")

        self.name = "SASRec"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.model = None
        self.num_users = None
        self.num_items = None
        self.user_sequences = {}

    def fit(self, R, progress_callback=None, visualizer = None):
        self.num_users, self.num_items = R.shape

        # Create sequences for prediction later
        for u in range(self.num_users):
            seq = R[u, :].nonzero()[0]
            # Add 1 to all item IDs because 0 is our padding index
            self.user_sequences[u] = seq + 1

        dataset = SeqDataset(R + 1, self.max_len) # Add 1 to item IDs
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SASRecModel(
            num_items=self.num_items, k=self.k, num_blocks=self.num_blocks,
            num_heads=self.num_heads, dropout_rate=self.dropout_rate, max_len=self.max_len - 1
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding

        self.model.train()
        for epoch in range(self.epochs):
            for seq, target in dataloader:
                optimizer.zero_grad()
                # Reshape for cross-entropy: (N, C), (N)
                logits = self.model(seq).view(-1, self.k)
                labels = target.view(-1)

                # Get scores by multiplying with item embeddings
                item_embeds = self.model.item_embedding.weight
                scores = torch.matmul(logits, item_embeds.transpose(0, 1))

                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

            if progress_callback:
                progress_callback((epoch + 1) / self.epochs)
        return self

    def predict(self):
        self.model.eval()
        predictions = np.zeros((self.num_users, self.num_items))

        with torch.no_grad():
            for u in range(self.num_users):
                seq = self.user_sequences[u]
                if len(seq) == 0:
                    continue

                # Prepare input sequence for prediction
                input_seq = np.zeros(self.max_len - 1, dtype=np.int64)
                end = min(len(seq), self.max_len - 1)
                input_seq[-end:] = seq[-end:]
                input_tensor = torch.LongTensor(input_seq).unsqueeze(0)

                # Get scores for all items
                scores = self.model.predict(input_tensor).squeeze(0)
                # Subtract 1 to map back to original item indices
                predictions[u, :] = scores.numpy()[1:]

        self.R_predicted = predictions
        return self.R_predicted
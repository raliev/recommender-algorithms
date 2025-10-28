import numpy as np
from .base import Recommender
import streamlit as st

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# --- SASRecModel class (no changes needed) ---
class SASRecModel(nn.Module):
    # ... (keep existing code)
    def __init__(self, num_items, k, num_blocks, num_heads, dropout_rate, max_len):
        super(SASRecModel, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 1, k, padding_idx=0) # +1 for padding
        self.positional_embedding = nn.Embedding(max_len, k)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=k, nhead=num_heads, dim_feedforward=k * 4, dropout=dropout_rate, batch_first=True, activation=nn.GELU()), # Use GELU activation
            num_layers=num_blocks
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(k, eps=1e-6)

    def forward(self, seq):
        # seq shape: (batch_size, seq_len)
        seq_len = seq.size(1)
        # Ensure positions are within bounds of positional embedding layer
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).clamp(max=self.positional_embedding.num_embeddings - 1)


        # Embeddings and positional encoding
        # Use layer norm before adding positional embedding
        x = self.layer_norm(self.item_embedding(seq))
        x += self.positional_embedding(positions)
        x = self.dropout(x)

        # Causal mask to prevent attending to future items
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=seq.device)

        # Self-attention encoder
        log_feats = self.encoder(x, mask=mask, is_causal=True) # Specify causal for newer PyTorch versions

        return log_feats

    def predict(self, seq):
        log_feats = self.forward(seq) # (batch, seq_len, k)
        final_feat = log_feats[:, -1, :] # Use the output of the last item in the sequence

        # Dot product with all item embeddings (excluding padding) to get scores
        item_embeds = self.item_embedding.weight[1:] # Exclude padding idx 0
        logits = torch.matmul(final_feat, item_embeds.transpose(0, 1))
        return logits

# --- SeqDataset class (minor adjustment for target) ---
class SeqDataset(Dataset):
    def __init__(self, R_sequences, max_len):
        # R_sequences should be a dictionary {user_idx: [item_idx_1, item_idx_2,...]}
        # item_idxs should already be shifted by +1 (0 reserved for padding)
        self.users = list(R_sequences.keys())
        self.user_items = R_sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_items[user]

        # Pad sequence and create input/target
        padded_seq = np.zeros(self.max_len, dtype=np.int64)
        input_seq = np.zeros(self.max_len, dtype=np.int64)
        target_seq = np.zeros(self.max_len, dtype=np.int64)

        # The actual sequence length used (<= max_len)
        seq_len = min(len(seq), self.max_len)

        if seq_len > 0:
            padded_seq[-seq_len:] = seq[-seq_len:]

            # Input: sequence up to the second-to-last item
            # Target: sequence shifted left (predict next item)
            input_seq[-seq_len:] = padded_seq[-seq_len:]
            target_seq[-(seq_len-1):] = padded_seq[-(seq_len-1):] # Target starts from the second item


        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)


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
        self.user_sequences = {} # Store original sequences for prediction

    def fit(self, R, progress_callback=None, visualizer = None):
        self.num_users, self.num_items = R.shape

        # Create sequences dictionary {user_idx: shifted_item_indices}
        fit_sequences = {}
        for u in range(self.num_users):
            seq = R[u, :].nonzero()[0]
            # Add 1 to all item IDs because 0 is our padding index
            shifted_seq = seq + 1
            if len(shifted_seq) > 0:
                fit_sequences[u] = shifted_seq
                # Store original (shifted) sequence for prediction
                self.user_sequences[u] = shifted_seq


        if not fit_sequences:
            st.warning("No sequences found in the training data for SASRec.")
            return self


        dataset = SeqDataset(fit_sequences, self.max_len) # Pass sequences dict
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SASRecModel(
            num_items=self.num_items, k=self.k, num_blocks=self.num_blocks,
            num_heads=self.num_heads, dropout_rate=self.dropout_rate, max_len=self.max_len # Pass max_len for pos embed
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)) # Use Adam params from paper
        # Use CrossEntropyLoss for next item prediction
        criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding

        if visualizer:
            params_to_save = {
                'algorithm': self.name, 'k': self.k, 'epochs_set': self.epochs,
                'learning_rate': self.learning_rate, 'batch_size': self.batch_size,
                'max_len': self.max_len, 'num_blocks': self.num_blocks,
                'num_heads': self.num_heads, 'dropout_rate': self.dropout_rate
            }
            visualizer.k_factors = self.k # Inform visualizer about embedding dim
            visualizer.start_run(params_to_save)


        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            for seq, target in dataloader:
                optimizer.zero_grad()

                # Get sequence embeddings from the model
                log_feats = self.model(seq) # (batch, max_len, k)

                # Predict next item scores: dot product with item embeddings
                # We need scores for all possible next items (excluding padding item 0)
                item_embeds = self.model.item_embedding.weight[1:] # (num_items, k)

                # Reshape log_feats to (batch * max_len, k) for matmul
                log_feats_flat = log_feats.view(-1, self.k)

                # Calculate scores for all items
                scores = torch.matmul(log_feats_flat, item_embeds.transpose(0, 1)) # (batch * max_len, num_items)

                # Reshape target to (batch * max_len)
                # Target IDs need to be adjusted: 0 for padding, 1 to num_items for actual items
                # Shift target indices down by 1 so they match the scores (which exclude item 0)
                labels = target.view(-1)
                valid_labels_mask = labels > 0 # Mask out padding

                if not valid_labels_mask.any(): # Skip batch if no valid labels
                    continue

                adjusted_labels = (labels[valid_labels_mask] - 1).long() # Adjust valid labels
                valid_scores = scores.view(target.shape[0], self.max_len, -1)[valid_labels_mask.view(target.shape[0], self.max_len)]


                loss = criterion(valid_scores, adjusted_labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0

            if visualizer:
                # SASRec doesn't have explicit P/Q matrices like MF models.
                # We can pass the item embedding matrix as a substitute for 'Q'
                # and maybe None or zeros for 'P' for the snapshot plotter.
                q_embed = self.model.item_embedding.weight.data.cpu().numpy()
                visualizer.record_iteration(
                    iteration_num=epoch + 1,
                    total_iterations=self.epochs,
                    objective=avg_epoch_loss,
                    Q=q_embed, # Pass item embeddings
                    P=None # No direct user embedding matrix
                    # No factor change for SASRec
                )

            if progress_callback:
                progress_callback((epoch + 1) / self.epochs)

        if visualizer:
            visualizer.end_run()

        return self

    def predict(self):
        self.model.eval()
        predictions = np.zeros((self.num_users, self.num_items))

        with torch.no_grad():
            for u in range(self.num_users):
                seq = self.user_sequences.get(u) # Use stored sequence
                if seq is None or len(seq) == 0:
                    continue

                # Prepare input sequence for prediction (needs max_len)
                input_seq = np.zeros(self.max_len, dtype=np.int64)
                # Use the last max_len items
                end = min(len(seq), self.max_len)
                input_seq[-end:] = seq[-end:]
                input_tensor = torch.LongTensor(input_seq).unsqueeze(0)

                # Get scores for all items (these are logits)
                scores = self.model.predict(input_tensor).squeeze(0)
                # Scores correspond to item indices 1 to num_items
                predictions[u, :] = scores.numpy() # Fill the prediction matrix

        self.R_predicted = predictions
        return self.R_predicted
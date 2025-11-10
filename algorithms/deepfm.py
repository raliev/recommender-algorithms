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

class DeepFMModel(nn.Module):
    """
    A PyTorch implementation of DeepFM for the user-item-only context.
    Combines Linear (1st order) + FM (2nd order) + MLP (high order).
    """
    def __init__(self, num_users, num_items, k):
        super(DeepFMModel, self).__init__()

        # Shared Embeddings
        self.user_embedding = nn.Embedding(num_users, k)
        self.item_embedding = nn.Embedding(num_items, k)

        # 1. Linear (1st Order) Component
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # 2. FM (2nd Order) Component
        # This is computed via the dot product of the shared embeddings

        # 3. Deep (MLP) Component
        mlp_input_dim = 2 * k
        self.mlp_layers = nn.Sequential(
            nn.Linear(mlp_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.mlp_final_layer = nn.Linear(16, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, item_indices):
        # Get Embeddings
        e_u = self.user_embedding(user_indices)
        e_i = self.item_embedding(item_indices)

        # --- Linear Component ---
        b_u = self.user_bias(user_indices)
        b_i = self.item_bias(item_indices)
        y_linear = b_u + b_i + self.global_bias

        # --- FM Component ---
        # (e_u * e_i) is element-wise product, .sum() is dot product
        y_fm = (e_u * e_i).sum(dim=1, keepdim=True)

        # --- Deep Component ---
        mlp_input = torch.cat([e_u, e_i], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        y_mlp = self.mlp_final_layer(mlp_output)

        # --- Final Prediction ---
        # Sum all components
        logits = y_linear + y_fm + y_mlp

        return torch.sigmoid(logits.squeeze(1))


class DeepFMRecommender(Recommender):
    def __init__(self, k, epochs=10, batch_size=64, learning_rate=0.001, lambda_reg=0.01, **kwargs):
        super().__init__(k)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch' to use this recommender.")

        self.name = "DeepFM"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.model = None
        self.num_users = None
        self.num_items = None

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.num_users, self.num_items = R.shape

        # Prepare training data (same as NCF: 1 positive, 4 negatives)
        user_ids, item_ids = R.nonzero()
        labels = np.ones(len(user_ids), dtype=np.float32)

        num_neg_samples = 4
        neg_user_ids, neg_item_ids = [], []
        rated_items_map = {u: set(R[u, :].nonzero()[0]) for u in range(self.num_users)}

        for u, items in rated_items_map.items():
            for _ in range(len(items) * num_neg_samples):
                j = np.random.randint(self.num_items)
                while j in items:
                    j = np.random.randint(self.num_items)
                neg_user_ids.append(u)
                neg_item_ids.append(j)

        all_user_ids = torch.LongTensor(np.concatenate([user_ids, np.array(neg_user_ids)]))
        all_item_ids = torch.LongTensor(np.concatenate([item_ids, np.array(neg_item_ids)]))
        all_labels = torch.FloatTensor(np.concatenate([labels, np.zeros(len(neg_user_ids), dtype=np.float32)]))

        dataset = TensorDataset(all_user_ids, all_item_ids, all_labels.view(-1, 1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = DeepFMModel(self.num_users, self.num_items, self.k)
        # Use weight_decay for L2 regularization (lambda_reg)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg)
        criterion = nn.BCELoss()

        if visualizer:
            visualizer.k_factors = self.k
            visualizer.start_run(params_to_save, R=R)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            for u, i, l in dataloader:
                optimizer.zero_grad()
                predictions = self.model(u, i)
                loss = criterion(predictions, l.squeeze(1).float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            if visualizer:
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
                current_iteration = epoch + 1

                # Pass embeddings for visualization
                P_embed = self.model.user_embedding.weight.data.cpu().numpy()
                Q_embed = self.model.item_embedding.weight.data.cpu().numpy()

                visualizer.record_iteration(
                    iteration_num=current_iteration,
                    total_iterations=self.epochs,
                    P=P_embed,
                    Q=Q_embed,
                    objective=avg_epoch_loss
                )
            if progress_callback:
                progress_callback((epoch + 1) / self.epochs)

        if visualizer:
            R_predicted_final = self.predict()
            visualizer.end_run(R_predicted_final=R_predicted_final)

        return self

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            all_user_ids = torch.arange(self.num_users).long()
            all_item_ids = torch.arange(self.num_items).long()

            users_grid, items_grid = torch.meshgrid(all_user_ids, all_item_ids, indexing='ij')
            user_input = users_grid.flatten()
            item_input = items_grid.flatten()

            # Predict in batches to avoid OOM errors on large matrices
            predictions = []
            with torch.no_grad():
                # Use a larger batch size for prediction
                pred_batch_size = self.batch_size * 4
                for i in range(0, len(user_input), pred_batch_size):
                    u_batch = user_input[i:i+pred_batch_size]
                    i_batch = item_input[i:i+pred_batch_size]
                    batch_preds = self.model(u_batch, i_batch)
                    predictions.append(batch_preds)

            predictions = torch.cat(predictions)
            self.R_predicted = predictions.view(self.num_users, self.num_items).numpy()

        return self.R_predicted
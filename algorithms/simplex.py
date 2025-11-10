import numpy as np
from .base import Recommender

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class SimpleXModel(nn.Module):
    """
    PyTorch module for SimpleX, a simple bi-encoder model
    trained with contrastive learning.

    """
    def __init__(self, num_users, num_items, k, dropout_rate=0.1):
        super(SimpleXModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, k)
        self.item_embedding = nn.Embedding(num_items, k)

        # Dropout for consistency regularization
        #
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        """
        Forward pass. Computes two versions of embeddings for
        consistency regularization.

        """
        # First pass (main)
        p_u = self.user_embedding(user_indices)
        q_i = self.item_embedding(item_indices)

        # Second pass (for consistency)
        # Dropout is applied during training
        p_u_prime = self.dropout(p_u)
        q_i_prime = self.dropout(q_i)

        return p_u, q_i, p_u_prime, q_i_prime

    def predict(self):
        """
        Generates the full score matrix for prediction.

        """
        return self.user_embedding.weight @ self.item_embedding.weight.T

class SimpleXRecommender(Recommender):
    def __init__(self, k, epochs=30, batch_size=256, learning_rate=0.001,
                 lambda_reg=0.1, tau=0.1, dropout_rate=0.1, **kwargs):
        super().__init__(k)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please run 'pip install torch' to use this recommender.")

        self.name = "SimpleX"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg # Corresponds to gamma (consistency)
        self.tau = tau # Temperature parameter
        self.dropout_rate = dropout_rate

        self.model = None
        self.num_users = None
        self.num_items = None

    def _calculate_infonce_loss(self, p_u, q_i, tau):
        """
        Calculates the InfoNCE loss using in-batch negatives.

        """
        # Calculate scores: user embeddings vs. all item embeddings in the batch
        # This creates a (batch_size, batch_size) similarity matrix
        scores = (p_u @ q_i.T) / tau #

        # The positive pairs are on the diagonal
        # We use log_softmax for numerical stability
        loss = F.cross_entropy(scores, torch.arange(scores.size(0), device=scores.device))
        return loss

    def _calculate_consistency_loss(self, p_u, q_i, p_u_prime, q_i_prime):
        """
        Calculates the consistency regularization loss.

        """
        loss_p = F.mse_loss(p_u, p_u_prime)
        loss_q = F.mse_loss(q_i, q_i_prime)
        return (loss_p + loss_q) / 2

    def fit(self, R, progress_callback=None, visualizer=None, params_to_save=None):
        self.num_users, self.num_items = R.shape

        # SimpleX uses only positive interactions for training
        user_ids, item_ids = R.nonzero()

        dataset = TensorDataset(torch.LongTensor(user_ids), torch.LongTensor(item_ids))
        # Drop last batch if it's smaller than 2 (causes issues with in-batch)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=(True if len(dataset) % self.batch_size < 2 else False))

        self.model = SimpleXModel(self.num_users, self.num_items, self.k, self.dropout_rate)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if visualizer:
            visualizer.k_factors = self.k
            visualizer.start_run(params_to_save, R=R)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            for u_batch_indices, i_batch_indices in dataloader:

                # Get embeddings (two passes for consistency reg)
                #
                p_u, q_i, p_u_prime, q_i_prime = self.model(u_batch_indices, i_batch_indices)

                # 1. InfoNCE Loss (Contrastive)
                #
                loss_infonce = self._calculate_infonce_loss(p_u, q_i, self.tau)

                # 2. Consistency Loss
                #
                loss_consistency = self._calculate_consistency_loss(p_u, q_i, p_u_prime, q_i_prime)

                # 3. Final Objective
                #
                loss = loss_infonce + self.lambda_reg * loss_consistency

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if visualizer:
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
                current_iteration = epoch + 1

                # Get embeddings for snapshot
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
            # Generate final predictions and pass to end_run for breakdown
            R_predicted_final = self.predict()
            visualizer.end_run(R_predicted_final=R_predicted_final)

        return self

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            self.R_predicted = self.model.predict().cpu().numpy()
        return self.R_predicted
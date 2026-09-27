'''
TimeSeriesTransformer: A Comprehensive Transformer Model for Time Series Prediction
==================================================================================

This module provides a complete transformer-based solution for time series forecasting
with the following features:
- Custom transformer architecture optimized for time series
- Multiple loss functions (MSE, MAE, Huber, Custom)
- Training, validation, and testing pipelines
- Greedy inference for autoregressive prediction
- Configurable model parameters
- Data preprocessing and augmentation

Classes:
--------
- TimeSeriesTransformer: Main transformer model
- TimeSeriesDataset: Dataset class for time series data
- TimeSeriesTrainer: Training pipeline with validation
- TimeSeriesPredictor: Inference and prediction utilities

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union, Dict, Any
import math
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for transformer.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series prediction.
    
    Parameters:
    -----------
    input_size : int
        Number of input features (M)
    output_size : int
        Number of output features (N)
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    d_ff : int
        Feed-forward dimension
    dropout : float
        Dropout rate
    max_seq_len : int
        Maximum sequence length
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 1000):
        
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass of the transformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        mask : torch.Tensor, optional
            Attention mask
        return_attention : bool
            Whether to return attention weights
            
        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_size)
        attention_weights : list, optional
            List of attention weights from each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Output projection
        output = self.output_projection(x)
        
        if return_attention:
            return output, attention_weights
        else:
            return output


class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series data of shape (L, M)
    TIN : int
        Input sequence length
    T : int
        Output sequence length
    stride : int
        Stride for creating sequences
    scaler : sklearn scaler, optional
        Scaler for normalization
    """
    
    def __init__(self, data, TIN, T, stride=1, scaler=None):
        self.data = torch.FloatTensor(data)
        self.TIN = TIN
        self.T = T
        self.stride = stride
        self.scaler = scaler
        
        # Calculate number of samples
        self.n_samples = (len(data) - TIN - T) // stride + 1
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.TIN + self.T
        
        sequence = self.data[start_idx:end_idx]
        
        # Split into input and target
        input_seq = sequence[:self.TIN]
        target_seq = sequence[self.TIN:self.TIN + self.T]
        
        return input_seq, target_seq


class LossFunctions:
    """
    Collection of loss functions for time series prediction.
    """
    
    @staticmethod
    def mse_loss(pred, target, mask=None):
        """Mean Squared Error loss."""
        if mask is not None:
            loss = F.mse_loss(pred * mask, target * mask, reduction='none')
            return loss.sum() / mask.sum()
        return F.mse_loss(pred, target)
    
    @staticmethod
    def mae_loss(pred, target, mask=None):
        """Mean Absolute Error loss."""
        if mask is not None:
            loss = F.l1_loss(pred * mask, target * mask, reduction='none')
            return loss.sum() / mask.sum()
        return F.l1_loss(pred, target)
    
    @staticmethod
    def huber_loss(pred, target, delta=1.0, mask=None):
        """Huber loss."""
        if mask is not None:
            loss = F.huber_loss(pred * mask, target * mask, delta=delta, reduction='none')
            return loss.sum() / mask.sum()
        return F.huber_loss(pred, target, delta=delta)
    
    @staticmethod
    def custom_loss(pred, target, alpha=0.5, mask=None):
        """
        Custom loss combining MSE and MAE.
        
        Parameters:
        -----------
        alpha : float
            Weight for MSE vs MAE (0 = pure MAE, 1 = pure MSE)
        """
        mse = LossFunctions.mse_loss(pred, target, mask)
        mae = LossFunctions.mae_loss(pred, target, mask)
        return alpha * mse + (1 - alpha) * mae


class TimeSeriesTrainer:
    """
    Training pipeline for the TimeSeriesTransformer model.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion, clip_grad=1.0):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(input_seq)
            
            # Calculate loss
            loss = criterion(output, target_seq)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward pass
                output = self.model(input_seq)
                
                # Calculate loss
                loss = criterion(output, target_seq)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs=100, 
              lr=0.001, 
              weight_decay=1e-5,
              patience=10,
              clip_grad=1.0,
              loss_type='mse',
              alpha=0.5):
        """
        Train the model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for optimizer
        patience : int
            Early stopping patience
        clip_grad : float
            Gradient clipping value
        loss_type : str
            Loss function type ('mse', 'mae', 'huber', 'custom')
        alpha : float
            Alpha parameter for custom loss
        """
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Setup loss function
        if loss_type == 'mse':
            criterion = LossFunctions.mse_loss
        elif loss_type == 'mae':
            criterion = LossFunctions.mae_loss
        elif loss_type == 'huber':
            criterion = lambda pred, target: LossFunctions.huber_loss(pred, target)
        elif loss_type == 'custom':
            criterion = lambda pred, target: LossFunctions.custom_loss(pred, target, alpha)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training with {loss_type} loss...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion, clip_grad)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping check
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
        
        return self.train_losses, self.val_losses
    
    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()


class TimeSeriesPredictor:
    """
    Prediction utilities for the TimeSeriesTransformer model.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def predict(self, input_seq, T_predict):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        input_seq : torch.Tensor
            Input sequence of shape (batch_size, TIN, M)
        T_predict : int
            Number of time steps to predict
            
        Returns:
        --------
        predictions : torch.Tensor
            Predictions of shape (batch_size, T_predict, N)
        """
        self.model.eval()
        
        with torch.no_grad():
            input_seq = input_seq.to(self.device)
            predictions = self.model(input_seq)
            
            # If we need more predictions than the model outputs
            if T_predict > predictions.shape[1]:
                # Use greedy inference for longer sequences
                predictions = self.greedy_inference(input_seq, T_predict)
        
        return predictions
    
    def greedy_inference(self, input_seq, T_predict):
        """
        Greedy inference for autoregressive prediction.
        
        Parameters:
        -----------
        input_seq : torch.Tensor
            Input sequence of shape (batch_size, TIN, M)
        T_predict : int
            Number of time steps to predict
            
        Returns:
        --------
        predictions : torch.Tensor
            Predictions of shape (batch_size, T_predict, N)
        """
        self.model.eval()
        
        batch_size = input_seq.shape[0]
        predictions = torch.zeros(batch_size, T_predict, self.model.output_size, device=self.device)
        
        with torch.no_grad():
            # Initial prediction
            current_input = input_seq.clone()
            
            for t in range(T_predict):
                # Get prediction for current time step
                output = self.model(current_input)
                pred_t = output[:, -1:, :]  # Take last prediction
                
                # Store prediction
                predictions[:, t:t+1, :] = pred_t
                
                # Update input sequence for next iteration
                if t < T_predict - 1:
                    # Shift input sequence and add new prediction
                    current_input = torch.cat([current_input[:, 1:, :], pred_t], dim=1)
        
        return predictions
    
    def evaluate(self, test_loader, metrics=['mse', 'mae']):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader
        metrics : list
            List of metrics to calculate
            
        Returns:
        --------
        results : dict
            Dictionary with metric values
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for input_seq, target_seq in test_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Get predictions
                predictions = self.model(input_seq)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_seq.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
        
        return results, predictions, targets
    
    def plot_predictions(self, predictions, targets, sample_idx=0, feature_idx=0):
        """
        Plot predictions vs targets for a specific sample and feature.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            True targets
        sample_idx : int
            Sample index to plot
        feature_idx : int
            Feature index to plot
        """
        plt.figure(figsize=(12, 6))
        
        pred_series = predictions[sample_idx, :, feature_idx]
        target_series = targets[sample_idx, :, feature_idx]
        
        plt.plot(target_series, label='True', linewidth=2)
        plt.plot(pred_series, label='Predicted', linewidth=2, linestyle='--')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'Predictions vs Targets (Sample {sample_idx}, Feature {feature_idx})')
        plt.legend()
        plt.grid(True)
        plt.show()


def create_data_loaders(data, TIN, T, train_ratio=0.7, val_ratio=0.15, 
                       batch_size=32, shuffle=True, scaler=None):
    """
    Create train, validation, and test data loaders.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series data
    TIN : int
        Input sequence length
    T : int
        Output sequence length
    train_ratio : float
        Ratio of data for training
    val_ratio : float
        Ratio of data for validation
    batch_size : int
        Batch size for data loaders
    shuffle : bool
        Whether to shuffle the data
    scaler : sklearn scaler, optional
        Scaler for normalization
        
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader
        Data loaders for training, validation, and testing
    """
    
    # Split data
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, TIN, T, scaler=scaler)
    val_dataset = TimeSeriesDataset(val_data, TIN, T, scaler=scaler)
    test_dataset = TimeSeriesDataset(test_data, TIN, T, scaler=scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main_example():
    """
    Example usage of the TimeSeriesTransformer model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample data
    L, M = 1000, 5  # 1000 time steps, 5 features
    np.random.seed(42)
    data = np.random.randn(L, M)
    
    # Add some temporal structure
    for i in range(M):
        data[:, i] += np.sin(np.linspace(0, 4*np.pi, L)) * (i+1) * 0.5
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN = 50  # Input sequence length
    T = 10    # Output sequence length
    N = M     # Number of output features (same as input for this example)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=32
    )
    
    # Create model
    model = TimeSeriesTransformer(
        input_size=M,
        output_size=N,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    )
    
    # Create trainer
    trainer = TimeSeriesTrainer(model, device)
    
    # Train model
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=50,
        lr=0.001,
        patience=10,
        loss_type='mse'
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Create predictor
    predictor = TimeSeriesPredictor(model, device)
    
    # Evaluate model
    results, predictions, targets = predictor.evaluate(test_loader)
    print("Test Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    # Plot some predictions
    predictor.plot_predictions(predictions, targets, sample_idx=0, feature_idx=0)
    
    # Test greedy inference
    sample_input = torch.FloatTensor(data_normalized[:TIN]).unsqueeze(0)
    greedy_predictions = predictor.greedy_inference(sample_input, T_predict=20)
    print(f"Greedy inference shape: {greedy_predictions.shape}")


if __name__ == "__main__":
    main_example()

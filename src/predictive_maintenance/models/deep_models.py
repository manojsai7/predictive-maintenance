"""Deep learning models for time-series predictive maintenance."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import joblib


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series sensor data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            labels: Array of shape (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    """LSTM model for RUL prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for RUL prediction."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling over sequence
        pooled = torch.mean(transformer_out, dim=1)
        
        # Fully connected layers
        out = self.dropout(pooled)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class DeepRULPredictor:
    """Deep learning based RUL predictor."""
    
    def __init__(self, model_type: str = 'lstm', input_size: int = 10,
                 sequence_length: int = 50, device: Optional[str] = None):
        """
        Initialize deep RUL predictor.
        
        Args:
            model_type: Type of model ('lstm' or 'transformer')
            input_size: Number of input features
            sequence_length: Length of input sequences
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def build_model(self, **model_params):
        """Build the deep learning model."""
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=model_params.get('hidden_size', 64),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_size=self.input_size,
                d_model=model_params.get('d_model', 64),
                nhead=model_params.get('nhead', 4),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.2)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        
        learning_rate = model_params.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def prepare_sequences(self, data: np.ndarray, labels: np.ndarray,
                         sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for training.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            labels: Labels of shape (n_samples,)
            sequence_length: Length of sequences to create
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            sequence_labels.append(labels[i + sequence_length - 1])
        
        return np.array(sequences), np.array(sequence_labels)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
             batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the deep learning model.
        
        Args:
            X: Training sequences of shape (n_samples, sequence_length, n_features)
            y: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            self.build_model()
        
        # Split into train and validation
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL for new sequences.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'sequence_length': self.sequence_length
        }
        torch.save(model_dict, path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        model_dict = torch.load(path, map_location=self.device)
        
        self.model_type = model_dict['model_type']
        self.input_size = model_dict['input_size']
        self.sequence_length = model_dict['sequence_length']
        
        self.build_model()
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])

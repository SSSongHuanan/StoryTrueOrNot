import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        """Multi-head attention mechanism
        
        Args:
            hidden_size: Size of hidden layer
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        assert hidden_size % num_heads == 0
        self.head_size = hidden_size // num_heads
        
        # Linear projections for Query, Key, Value
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """Forward pass of multi-head attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
            
        Returns:
            Attention output of shape (batch_size, seq_length, hidden_size)
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Linear transformations and reshape
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_size)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_size)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_size)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return self.out_linear(out)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        """Residual block with feed-forward network
        
        Args:
            hidden_size: Size of hidden layer
            dropout_rate: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        return self.norm(x + self.layer(x))

class StoryClassifier(nn.Module):
    def __init__(self, input_size, config):
        """Story classifier model with attention mechanism
        
        Args:
            input_size: Dimension of input features
            config: Model configuration dictionary
        """
        super(StoryClassifier, self).__init__()
        
        # Configuration validation and defaults
        if isinstance(config, dict) and 'model' in config:
            hidden_size = config['model'].get('hidden_size', 768)
            dropout_rate = config['model'].get('dropout_rate', 0.1)
        else:
            logger.warning("Invalid config format, using default values")
            hidden_size = 768
            dropout_rate = 0.1
        
        # Feature projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate)
            for _ in range(4)
        ])
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(hidden_size, num_heads=8)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Global feature extraction
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x):
        """Forward pass of the classifier
        
        Args:
            x: Input features
            
        Returns:
            Binary classification probability
        """
        # Feature projection
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply residual layers
        for residual_layer in self.residual_layers:
            x = residual_layer(x)
        
        # Apply attention mechanism
        attended = self.attention(x)
        x = self.attention_norm(x + attended)
        
        # Global feature pooling
        x = x.transpose(1, 2)
        x = self.global_features(x)
        
        # Classification
        logits = self.classifier(x)
        return torch.sigmoid(logits)

class ModelEnsemble(nn.Module):
    def __init__(self, input_size, config):
        """Ensemble of story classifiers
        
        Args:
            input_size: Dimension of input features
            config: Model configuration dictionary
        """
        super(ModelEnsemble, self).__init__()
        
        # Configuration validation and defaults
        if isinstance(config, dict) and 'model' in config:
            self.num_models = config['model'].get('num_models', 3)
        else:
            logger.warning("Invalid config format, using default num_models=3")
            self.num_models = 3
        
        # Create multiple classifier instances
        self.models = nn.ModuleList([
            StoryClassifier(input_size, config)
            for _ in range(self.num_models)
        ])
        
        # Model weights for weighted averaging
        self.model_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
    def forward(self, x):
        """Forward pass of the ensemble
        
        Args:
            x: Input features
            
        Returns:
            Weighted average of model predictions
        """
        # Get predictions from each model
        predictions = torch.stack([model(x) for model in self.models], dim=1)
        
        # Apply softmax to get normalized weights
        weights = F.softmax(self.model_weights, dim=0)
        
        # Weighted average of predictions
        ensemble_pred = torch.sum(predictions * weights.view(1, -1, 1), dim=1)
        
        return ensemble_pred

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """Focal Loss for binary classification
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """Calculate focal loss
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
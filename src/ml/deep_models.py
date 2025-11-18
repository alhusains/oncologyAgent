"""Deep learning model wrappers for survival analysis"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from pycox.models import CoxPH
    from pycox.preprocessing.label_transforms import LabTransCoxTime
    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False
    # Don't warn on import - only when trying to use the model
    torch = None
    nn = None


class DeepSurvWrapper:
    """
    Scikit-learn compatible wrapper for DeepSurv (Cox PH neural network).
    
    Uses pycox library for deep learning-based survival analysis.
    Compatible with scikit-survival structured arrays.
    """
    
    def __init__(
        self,
        num_nodes: Union[List[int], Tuple[int, ...]] = (64, 64),
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        batch_size: int = 128,
        epochs: int = 100,
        batch_norm: bool = True,
        output_bias: bool = False,
        device: Optional[str] = None,
        verbose: int = 0
    ):
        """
        Initialize DeepSurv wrapper.
        
        Args:
            num_nodes: List/tuple of hidden layer sizes (e.g., [64, 64])
            dropout: Dropout probability for regularization
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            batch_norm: Whether to use batch normalization
            output_bias: Whether to use bias in output layer
            device: Device to train on ('cuda' or 'cpu'). Auto-detects if None.
            verbose: Verbosity level (0=silent, 1=progress bar)
        """
        if not PYCOX_AVAILABLE:
            raise ImportError("pycox is required for DeepSurv. Install with: pip install pycox")
        
        # Store parameters exactly as passed for scikit-learn clone compatibility
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_norm = batch_norm
        self.output_bias = output_bias
        self.verbose = verbose
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model_ = None
        self.labtrans_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
    def _create_network(self, in_features: int) -> nn.Module:
        """Create the neural network architecture."""
        layers = []
        prev_size = in_features
        
        # Convert num_nodes to list for iteration (could be tuple or list)
        num_nodes_list = list(self.num_nodes) if isinstance(self.num_nodes, (list, tuple)) else [self.num_nodes]
        
        for hidden_size in num_nodes_list:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1, bias=self.output_bias))
        
        return nn.Sequential(*layers)
    
    def _prepare_data(self, X, y):
        """
        Prepare data for pycox.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Structured array with 'event' and 'time' fields (scikit-survival format)
        
        Returns:
            Tuple of (X_tensor, durations, events)
        """
        # Convert X to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        # Extract durations and events from structured array
        if isinstance(y, np.ndarray) and y.dtype.names:
            # scikit-survival structured array format
            events = y['event'].astype(np.float32)
            durations = y['time'].astype(np.float32)
        elif isinstance(y, (tuple, list)) and len(y) == 2:
            # Tuple format (events, durations)
            events, durations = y
            events = np.asarray(events, dtype=np.float32)
            durations = np.asarray(durations, dtype=np.float32)
        else:
            raise ValueError(
                "y must be a structured array with 'event' and 'time' fields, "
                "or a tuple of (events, durations)"
            )
        
        return X, durations, events
    
    def fit(self, X, y):
        """
        Fit the DeepSurv model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Structured array with 'event' and 'time' fields
        
        Returns:
            self
        """
        X, durations, events = self._prepare_data(X, y)
        
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        
        # Create network
        net = self._create_network(self.n_features_in_)
        
        # Initialize model
        self.model_ = CoxPH(net, torch.optim.Adam, device=self.device)
        
        # Set learning rate
        self.model_.optimizer.param_groups[0]['lr'] = self.learning_rate
        
        # Label transformation (required by pycox)
        self.labtrans_ = LabTransCoxTime()
        y_train = self.labtrans_.fit_transform(durations, events)
        
        # Training
        try:
            # Create data loader
            verbose = self.verbose > 0
            
            # Train the model
            self.model_.fit(
                X, 
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=verbose,
                val_data=None  # Could add validation data if available
            )
        except Exception as e:
            warnings.warn(f"DeepSurv training encountered an issue: {str(e)}")
            # Still return self for compatibility, but model may not be optimal
        
        return self
    
    def predict(self, X):
        """
        Predict risk scores (higher = higher risk).
        
        Args:
            X: Feature matrix
        
        Returns:
            Risk scores (negative log-hazard)
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        # Get predictions
        with torch.no_grad():
            risk_scores = self.model_.predict(X)
        
        # Return negative log-hazard (compatible with concordance index)
        # In Cox model, higher values = higher risk
        return risk_scores.flatten()
    
    def predict_survival_function(self, X, times=None):
        """
        Predict survival function (not fully implemented - placeholder).
        
        Args:
            X: Feature matrix
            times: Time points for prediction
        
        Returns:
            Placeholder - returns risk scores
        """
        warnings.warn("predict_survival_function not fully implemented for DeepSurv. Returning risk scores.")
        return self.predict(X)
    
    def score(self, X, y):
        """
        Calculate concordance index on test data.
        
        Args:
            X: Feature matrix
            y: Structured array with 'event' and 'time' fields
        
        Returns:
            Concordance index (C-index)
        """
        from sksurv.metrics import concordance_index_censored
        
        X, durations, events = self._prepare_data(X, y)
        risk_scores = self.predict(X)
        
        # Calculate C-index
        c_index = concordance_index_censored(
            events.astype(bool),
            durations,
            risk_scores
        )[0]
        
        return c_index
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'num_nodes': self.num_nodes,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'batch_norm': self.batch_norm,
            'output_bias': self.output_bias,
            'device': self.device,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        return self



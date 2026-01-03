# API Reference Documentation

## Overview

This document describes the public API of the "Learning Causality, Not Correlation" project. All modules follow a consistent design pattern and are fully documented.

## 1. Core Modules

### 1.1 Data Generation API

#### `DataGenerator` Class
```python
class DataGenerator:
    """
    Generate synthetic data with causal and spurious variables.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (see configs/data_config.yaml)
    seed : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, config, seed=42):
        pass
    
    def generate_environment(self, env_name, n_samples):
        """
        Generate data for a specific environment.
        
        Parameters
        ----------
        env_name : str
            Environment name ('E1', 'E2', 'E_test')
        n_samples : int
            Number of samples to generate
        
        Returns
        -------
        X : pandas.DataFrame
            Feature matrix with columns:
            - 'income' (causal)
            - 'credit_history' (causal)
            - 'application_channel' (spurious)
            - 'processing_day' (spurious)
        y : pandas.Series
            Binary target variable (0: no default, 1: default)
        """
        pass
    
    def generate_all_environments(self):
        """
        Generate all environments defined in configuration.
        
        Returns
        -------
        environments : dict
            Dictionary mapping environment names to (X, y) tuples
        """
        pass
```

DataProcessor Class

```python
class DataProcessor:
    """
    Process and split generated data.
    """
    
    def train_test_split(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        test_size : float, default=0.2
            Proportion of data for test set
        stratify : bool, default=True
            Whether to stratify by target variable
        
        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
            Split datasets
        """
        pass
```

1.2 Model API

Base Model Interface

```python
class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    
    @abstractmethod
    def fit(self, X, y, environments=None):
        """
        Train the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        environments : array-like of shape (n_samples,), optional
            Environment indices for each sample
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        pass
```

ModelFactory Class

```python
class ModelFactory:
    """
    Factory for creating model instances.
    """
    
    @staticmethod
    def create_model(model_name, config=None, **kwargs):
        """
        Create a model instance by name.
        
        Parameters
        ----------
        model_name : str
            Name of the model to create. Options:
            - 'logistic_regression'
            - 'random_forest'
            - 'mlp_erm'
            - 'irm'
            - 'vrex'
        config : dict, optional
            Model configuration
        **kwargs : dict
            Additional model-specific parameters
        
        Returns
        -------
        model : BaseModel
            Model instance
        """
        pass
```

1.3 Training API

BaseTrainer Class

```python
class BaseTrainer(ABC):
    """
    Base trainer class.
    """
    
    def __init__(self, model, config):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : BaseModel
            Model to train
        config : dict
            Training configuration
        """
        pass
    
    @abstractmethod
    def train(self, train_data, val_data=None):
        """
        Train the model.
        
        Parameters
        ----------
        train_data : tuple or DataLoader
            Training data (X, y) or DataLoader
        val_data : tuple or DataLoader, optional
            Validation data
        
        Returns
        -------
        history : dict
            Training history with metrics
        """
        pass
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        """
        pass
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        """
        pass
```

InvariantTrainer Class

```python
class InvariantTrainer(BaseTrainer):
    """
    Trainer for invariant learning models (IRM, V-REx).
    """
    
    def train(self, train_data, val_data=None):
        """
        Train with environment-aware loss.
        
        Parameters
        ----------
        train_data : dict
            Dictionary mapping environment names to (X, y) tuples
        val_data : dict, optional
            Validation data in same format
        
        Returns
        -------
        history : dict
            Training history with per-environment metrics
        """
        pass
```

1.4 Evaluation API

Evaluator Class

```python
class Evaluator:
    """
    Evaluate models on multiple metrics.
    """
    
    def __init__(self, metrics=None):
        """
        Initialize evaluator with metrics.
        
        Parameters
        ----------
        metrics : list of str, optional
            Metrics to compute. Default includes:
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'generalization_gap'
            - 'stability_score'
        """
        pass
    
    def evaluate(self, model, datasets, environments=None):
        """
        Evaluate model on multiple datasets.
        
        Parameters
        ----------
        model : BaseModel
            Trained model
        datasets : dict
            Dictionary mapping dataset names to (X, y) tuples
        environments : dict, optional
            Environment information for each dataset
        
        Returns
        -------
        results : pandas.DataFrame
            Evaluation results with columns:
            - dataset: Dataset name
            - metric: Metric name
            - value: Metric value
        """
        pass
    
    def compare_models(self, models, datasets):
        """
        Compare multiple models.
        
        Parameters
        ----------
        models : dict
            Dictionary mapping model names to BaseModel instances
        datasets : dict
            Dictionary mapping dataset names to (X, y) tuples
        
        Returns
        -------
        comparison : pandas.DataFrame
            Comparison results
        """
        pass
```

FeatureImportanceAnalyzer Class

```python
class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using SHAP.
    """
    
    def __init__(self, model, background_data=None):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        model : BaseModel
            Model to analyze
        background_data : array-like, optional
            Background data for SHAP estimation
        """
        pass
    
    def compute_shap_values(self, X):
        """
        Compute SHAP values.
        
        Parameters
        ----------
        X : array-like
            Input data
        
        Returns
        -------
        shap_values : array-like
            SHAP values with shape (n_samples, n_features)
        """
        pass
    
    def plot_summary(self, X, feature_names=None):
        """
        Plot SHAP summary plot.
        """
        pass
```

1.5 Visualization API

VisualizationManager Class

```python
class VisualizationManager:
    """
    Create visualizations for analysis.
    """
    
    def plot_performance_comparison(self, results_df, save_path=None):
        """
        Plot model performance comparison.
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            Results from Evaluator.compare_models()
        save_path : str, optional
            Path to save figure
        """
        pass
    
    def plot_feature_importance(self, importance_df, save_path=None):
        """
        Plot feature importance comparison.
        """
        pass
    
    def plot_latent_space(self, model, datasets, method='tsne', save_path=None):
        """
        Visualize latent space representations.
        
        Parameters
        ----------
        model : BaseModel
            Model with encoder (e.g., IRM model)
        datasets : dict
            Dictionary mapping environment names to data
        method : str, default='tsne'
            Dimensionality reduction method ('tsne' or 'pca')
        """
        pass
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history.
        """
        pass
```

2. Configuration API

ConfigLoader Class

```python
class ConfigLoader:
    """
    Load and manage configuration files.
    """
    
    @staticmethod
    def load_config(config_path):
        """
        Load YAML configuration file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file
        
        Returns
        -------
        config : dict
            Configuration dictionary
        """
        pass
    
    @staticmethod
    def merge_configs(base_config, override_config):
        """
        Merge multiple configurations.
        """
        pass
    
    @staticmethod
    def validate_config(config, schema_name):
        """
        Validate configuration against schema.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
        schema_name : str
            Schema name ('data', 'model', 'experiment', 'evaluation')
        """
        pass
```


# System Architecture

## Overview

The "Learning Causality, Not Correlation" project follows a modular, extensible architecture designed for reproducibility, scalability, and clarity. This document describes the high-level architecture and design decisions.

## 1. Architectural Principles

### 1.1 Design Goals
- **Reproducibility**: All experiments should be exactly reproducible
- **Modularity**: Components should be independent and interchangeable
- **Extensibility**: Easy to add new models, metrics, or datasets
- **Documentation**: Self-documenting code and configurations
- **Performance**: Efficient while maintaining readability

### 1.2 Key Decisions
- **Configuration-driven**: All parameters in YAML files
- **Type hints**: Full type annotation for better tooling support
- **Testing**: Comprehensive test coverage
- **Logging**: Structured logging throughout

## 2. High-Level Architecture

```

┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │   CLI API   │  │  Notebooks  │  │  Python API       │  │
│  └─────────────┘  └─────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                       │
│               ┌─────────────────────────┐                  │
│               │   ExperimentManager     │                  │
│               └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ Data Layer  │  │ Model Layer │  │ Evaluation Layer  │  │
│  └─────────────┘  └─────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │  Config     │  │  Logging    │  │  Utils & Helpers  │  │
│  └─────────────┘  └─────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘

```

## 3. Layer Details

### 3.1 User Interface Layer

#### Command Line Interface (CLI)
```python
# Entry point: src/main.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['generate', 'train', 'evaluate', 'experiment'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str)
    # ... additional arguments
```

Jupyter Notebooks

· notebooks/00_data_exploration.ipynb: Data exploration and validation
· notebooks/01_data_generation.ipynb: Interactive data generation
· notebooks/02_baseline_experiments.ipynb: Baseline model experiments
· notebooks/03_invariant_learning.ipynb: IRM implementation and training
· notebooks/04_comparative_analysis.ipynb: Results comparison
· notebooks/05_ablation_studies.ipynb: Ablation studies
· notebooks/06_visualization_dashboard.ipynb: Interactive dashboard
· notebooks/07_conclusions_insights.ipynb: Final analysis

Python API

· Direct import and usage of modules
· Designed for extensibility and integration

3.2 Orchestration Layer

ExperimentManager

· Coordinates entire experiment pipeline
· Handles configuration loading
· Manages resource allocation
· Tracks experiment state
· Produces final reports

3.3 Core Processing Layer

Data Layer

```
src/data_engineering/
├── data_generator.py          # Synthetic data generation
├── causal_mechanisms.py       # Causal relationships
├── spurious_correlations.py   # Spurious correlations
├── environment_simulator.py   # Environment creation
└── data_processor.py          # Data splitting and preprocessing
```

Data Flow:

```
Config → DataGenerator → Environments → DataProcessor → Train/Val/Test Splits
```

Model Layer

```
src/models/
├── baselines/                 # Traditional ML models
│   ├── sklearn_models.py      # LogisticRegression, RandomForest
│   ├── mlp_baseline.py        # Standard neural network
│   └── ensemble_baseline.py   # Ensemble methods
├── invariant_learning/        # Causal/invariant models
│   ├── invariant_encoder.py   # Shared encoder architecture
│   ├── irm_loss.py           # IRM loss implementation
│   ├── vrex_loss.py          # V-REx loss implementation
│   └── invariant_predictor.py # Invariant predictor
└── utils/                     # Model utilities
    ├── model_factory.py       # Factory pattern for models
    ├── checkpoint_manager.py  # Model serialization
    └── model_analyzer.py      # Model analysis tools
```

Training Flow:

```
ModelFactory → BaseModel → Trainer → Trained Model
```

Evaluation Layer

```
src/evaluation/
├── metrics.py                 # Standard and custom metrics
├── ood_metrics.py            # OOD-specific metrics
├── robustness_metrics.py     # Robustness measures
├── statistical_tests.py      # Statistical testing
├── hypothesis_testing.py     # Hypothesis validation
└── evaluator.py              # Main evaluation orchestrator
```

Evaluation Flow:

```
Trained Model + Test Data → Evaluator → Metrics → Statistical Analysis
```

3.4 Infrastructure Layer

Configuration Management

```
configs/
├── data_config.yaml           # Data generation parameters
├── model_configs.yaml         # Model architectures and hyperparameters
├── experiment_config.yaml     # Experiment setup
├── evaluation_protocol.yaml   # Evaluation metrics and procedures
└── project_config.yaml        # Global project settings
```

Logging System

· Structured logging with different levels (DEBUG, INFO, WARNING, ERROR)
· Log files per experiment run
· Console and file logging
· Experiment tracking

Utilities

```
src/utils/
├── file_utils.py             # File I/O operations
├── math_utils.py             # Mathematical utilities
├── statistical_utils.py      # Statistical functions
├── profiling.py              # Performance profiling
└── reproducibility.py        # Random seed management
```


Testing Architecture


Test Pyramid

```
        ┌─────────────────┐
        │   E2E Tests     │  (5%)
        └─────────────────┘
               │
        ┌─────────────────┐
        │ Integration     │  (15%)
        │   Tests         │
        └─────────────────┘
               │
        ┌─────────────────┐
        │   Unit Tests    │  (80%)
        └─────────────────┘
```

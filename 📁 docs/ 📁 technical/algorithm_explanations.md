# Algorithms and Methods Explanation

## Overview of Implemented Algorithms

This project implements and compares multiple algorithms for causal vs. correlational learning. Below are detailed explanations of each approach.

## 1. Baseline Algorithms (ERM-based)

### 1.1 Logistic Regression (scikit-learn)
- **Type**: Linear classification
- **Algorithm**: Maximum likelihood estimation with L2 regularization
- **Objective Function**: 
```

min_w C * Σ_i log(1 + exp(-y_i * (w^T x_i + b))) + ||w||_2^2

```
- **Implementation**: `sklearn.linear_model.LogisticRegression`
- **Expected Behavior**: Will learn both causal and spurious correlations due to ERM objective

### 1.2 Random Forest (scikit-learn)
- **Type**: Ensemble tree-based classification
- **Algorithm**: Bootstrap aggregating with decision trees
- **Key Parameters**: 
  - `n_estimators=100`: Number of trees
  - `max_depth=10`: Controls tree complexity
  - `min_samples_split=5`: Prevents overfitting
- **Feature Importance**: Gini impurity reduction
- **Expected Behavior**: May heavily use spurious features due to greedy splitting

### 1.3 Multilayer Perceptron (PyTorch)
- **Type**: Feedforward neural network
- **Architecture**: 4 → 64 → 32 → 1 with ReLU activations
- **Loss Function**: Binary Cross Entropy with Logits Loss
```

L = -[y·log(σ(logits)) + (1-y)·log(1-σ(logits))]

```
- **Optimizer**: Adam (β1=0.9, β2=0.999, ε=1e-8)
- **Expected Behavior**: Will learn complex correlations, potentially overfitting to spurious patterns

## 2. Invariant Learning Algorithms

### 2.1 Invariant Risk Minimization (IRM)
- **Source**: Arjovsky et al. (2019)
- **Core Idea**: Learn representation Φ where optimal classifier is invariant across environments
- **Objective Function**:
```

L = Σ_e [L_e(w ∘ Φ) + λ·||∇_{w|w=1.0} L_e(w·Φ)||^2]

```
  where:
  - `L_e`: Loss in environment e
  - `w`: Scalar "dummy" classifier
  - `λ`: Invariance penalty weight
- **Implementation Details**:
  - Encoder: 4 → 64 → 32 → 16 (latent representation)
  - Predictor: 16 → 32 → 1
  - Gradient penalty computed per environment

### 2.2 Variance Risk Extrapolation (V-REx) - Optional Extension
- **Source**: Krueger et al. (2021)
- **Objective Function**:
```

L = E_e[L_e] + β·Var_e(L_e)

```
- **Advantages**: More stable optimization than IRM
- **Implementation**: Available as alternative to IRM

## 3. Evaluation Algorithms

### 3.1 SHAP (SHapley Additive exPlanations)
- **Algorithm**: KernelSHAP approximation for model interpretability
- **Purpose**: Quantify feature importance for each prediction
- **Implementation**: `shap.KernelExplainer` for non-tree models
- **Output**: SHAP values showing contribution of each feature to prediction

### 3.2 Generalization Gap Calculation
- **Formula**:
```

Gap = Accuracy(E_test) - mean(Accuracy(E₁), Accuracy(E₂))

```
- **Statistical Test**: Paired t-test across multiple runs

### 3.3 Representation Similarity Analysis
- **Method**: Centered Kernel Alignment (CKA)
- **Purpose**: Measure similarity between representations across environments
- **Formula**:
```

CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

```
  where HSIC is Hilbert-Schmidt Independence Criterion

## 4. Statistical Analysis Methods

### 4.1 Hypothesis Testing Framework
- **Primary Test**: Welch's t-test for unequal variances
- **Multiple Testing Correction**: Bonferroni correction (α' = α/4)
- **Effect Size**: Cohen's d with 95% confidence intervals

### 4.2 Bootstrap Confidence Intervals
- **Method**: Percentile bootstrap with 1000 resamples
- **Purpose**: Estimate uncertainty in generalization gaps

## 5. Data Generation Algorithms

### 5.1 Synthetic Data Generation
- **Causal Variables**: Linear combination with Gaussian noise
- **Spurious Variables**: Environment-dependent linear mixing
- **Thresholding**: Percentile-based binary classification

### 5.2 Environment Simulation
- **Algorithm**: Controlled introduction of spurious correlations
- **Implementation**: Different linear mixing parameters per environment

## 6. Visualization Algorithms

### 6.1 t-SNE for Representation Visualization
- **Algorithm**: t-Distributed Stochastic Neighbor Embedding
- **Parameters**: perplexity=30, learning_rate=200, n_iter=1000
- **Purpose**: 2D visualization of high-dimensional representations

### 6.2 Partial Dependence Plots
- **Algorithm**: Marginal effect estimation
- **Purpose**: Visualize relationship between features and predictions


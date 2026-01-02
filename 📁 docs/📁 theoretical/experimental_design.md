
# Experimental Design

## 1. Objective of the Experiment

The objective of this project is to empirically demonstrate that:

1. **Traditional Empirical Risk Minimization (ERM) models** tend to learn spurious correlations and fail under out-of-distribution (OOD) settings.
2. **Invariant learning approaches (IRM-based models)** maintain performance under distribution shifts by focusing on causal mechanisms.

The experiment is designed to explicitly separate causal features from spurious features across multiple environments, allowing controlled evaluation of generalization behavior.

---

## 2. Variables and Definitions

### 2.1 Independent Variables (Features)

| Variable | Type | Distribution | Description |
|--------|------|-------------|-------------|
| `x1` (income) | Causal | Normal(μ = 50, σ = 10) | Applicant income |
| `x2` (credit_history) | Causal | Uniform(0, 100) | Credit history score |
| `s1` (application_channel) | Spurious | Environment-dependent | Application channel |
| `s2` (processing_day) | Spurious | Environment-dependent | Processing day |

Causal variables directly influence the target through an invariant mechanism.  
Spurious variables exhibit environment-specific correlations with the target.

---

### 2.2 Dependent Variable (Target)

**Target variable:**  
`y` (loan_default) ∈ {0, 1}

The target is generated as follows:

- Compute
y_score = 0.5 * x1 + 0.3 * x2 + ε


- Assign
y = 1 if y_score > Q3(y_score) y = 0 otherwise


- Noise term:
ε ~ Normal(0, 0.1)


This mechanism is invariant across all environments.

---

## 3. Distributional Environments

The data is generated under multiple environments to simulate distribution shifts.

| Environment | Time Period | Spurious Correlations | n_samples |
|------------|------------|----------------------|-----------|
| **E₁** | 2020–2021 | `s1` correlated with `y` (ρ = 0.8) | 5,000 |
| **E₂** | 2022–2023 | `s2` correlated with `y` (ρ = 0.7) | 5,000 |
| **E_test** | 2024 | No spurious correlations | 2,000 |

The test environment represents a true OOD scenario where spurious correlations disappear.

---

## 4. Data Generating Mechanisms

### 4.1 Causal Mechanism (Invariant)


y_score = 0.5 * x1 + 0.3 * x2 + ε
y = I(y_score > Q3(y_score))
This causal mechanism remains fixed across all environments.
4.2 Spurious Mechanisms (Environment-Specific)

Environment E₁:
s1 = 0.8 * y + 0.2 * Normal(0, 1)

Environment E₂:
s2 = 0.7 * y + 0.3 * Normal(0, 1)

Environment E_test:
s1, s2 ~ Normal(0, 1)
s1 ⟂ y , s2 ⟂ y
Spurious variables are predictive only in specific training environments.
5. Models Under Comparison
5.1 Group A: Baseline Models (ERM)
Logistic Regression
Random Forest
Multilayer Perceptron (MLP, 2 hidden layers)
These models minimize empirical risk without enforcing invariance across environments.
5.2 Group B: Invariant Models
Invariant Risk Minimization (IRM)
MLP with invariance regularization penalty
These models explicitly aim to learn environment-invariant representations.
6. Evaluation Metrics
6.1 Primary Metrics
Accuracy per environment: E₁, E₂, E_test
Generalization Gap:


acc(E_test) − mean(acc(E₁), acc(E₂))
Stability Score:


1 − std(accuracy across environments)
6.2 Secondary (Diagnostic) Metrics
Feature Importance: SHAP values per variable
Representation Similarity: Distance between learned embeddings
Causal Identification Rate:
Percentage of runs where causal variables (x1, x2) dominate feature importance
7. Experimental Protocol
Phase 1: Training
Train each model on environments {E₁, E₂}
Total training samples: 10,000
Validation: 20% holdout per environment
Hyperparameter tuning via grid search
Phase 2: Evaluation
Evaluate on held-out splits of E₁ and E₂ (in-distribution)
Evaluate on E_test (out-of-distribution)
Compute generalization and stability metrics
Phase 3: Analysis
Compare generalization gaps between ERM and IRM models
Analyze feature importance distributions
Visualize learned representations across environments
8. Success Criteria
Criterion 1: ERM Fragility
Success: ERM models exhibit >30% performance drop in E_test
Evidence: Significant degradation under OOD shift
Criterion 2: IRM Robustness
Success: IRM models show <10% generalization gap
Evidence: Stable accuracy across environments
Criterion 3: Causal Identification
Success: IRM models prioritize x1, x2 over s1, s2
Evidence: SHAP values consistently favor causal features
9. Assumptions and Limitations
Assumptions
Variables x1 and x2 are truly causal
Spurious correlations are strong enough to mislead ERM models
Environment shifts approximate real-world distribution changes
Limitations
Synthetic data may not capture full real-world complexity
IRM optimization can be unstable in practice
Limited number of causal variables (simplification)

---


# Learning Causality, Not Correlation

A research-oriented experimental framework to study why modern machine learning models fail under distribution shift and how invariant and causal-inspired methods can improve generalization beyond correlations.

---

## Overview

Modern machine learning and deep learning models excel at discovering statistical correlations in data. However, these correlations are often **spurious** and fail to generalize when the data distribution changes, a common scenario in real-world production systems.

This project provides a **reproducible experimental framework** to analyze this failure mode and to evaluate methods that aim to learn **stable, invariant representations** as a proxy for causal structure.

The goal is not to claim full causal discovery, but to rigorously study **robust generalization under distribution shift** using controlled synthetic environments and principled evaluation.

---

## Project Goals

This repository aims to:

1. Demonstrate why standard Empirical Risk Minimization (ERM) fails under distribution shift.
2. Study how models exploit spurious correlations and shortcut features.
3. Implement and evaluate invariant learning methods inspired by causal reasoning.
4. Provide a clean experimental setup for Out-of-Distribution (OOD) evaluation.
5. Bridge theoretical insights from causal ML with practical machine learning systems.

---

## Scientific Motivation

Empirical evidence shows that many ML failures in production are not due to lack of capacity, but due to **reliance on unstable correlations** that do not persist across environments.

From a causal perspective, these failures arise because models optimize prediction accuracy without distinguishing between:
- **Causal mechanisms** (stable across environments)
- **Spurious correlations** (environment-dependent)

This project studies this distinction explicitly.

---

## Core References (Foundational Papers)

The design and scope of this project are grounded in the following open-access papers:

1. **Invariant Risk Minimization**  
   Arjovsky et al., 2019  
   Introduces the formal framework for learning predictors that are invariant across multiple environments.  
   https://arxiv.org/abs/1907.02893

2. **Predicting with Confidence on Unseen Distributions**  
   Guillory et al., 2021  
   Provides methodology for evaluating model reliability under unseen distribution shifts.  
   https://arxiv.org/abs/2107.03315

3. **Causal Machine Learning: A Survey and Open Problems**  
   Schölkopf et al., 2022  
   Offers a comprehensive overview of causal methods applied to machine learning and their limitations.  
   https://arxiv.org/abs/2206.15475

4. **Shortcut Learning in Deep Neural Networks**  
   Geirhos et al., 2020  
   Demonstrates how deep models rely on brittle shortcuts instead of robust features.  
   https://arxiv.org/abs/2004.07780

5. **Distribution Shift and Robust Evaluation (Complementary Work)**  
   Related work on OOD benchmarks and robustness evaluation protocols.

These papers define the **conceptual and experimental boundaries** of the project.

---

## Problem Formulation

Let \( D_{train} \) be a training distribution composed of multiple environments \( \{E_1, E_2, \dots\} \), each sharing the same underlying causal mechanism but differing in spurious correlations.

Standard ERM optimizes:

\[
\min_f \mathbb{E}_{(x,y)\sim D_{train}}[L(f(x), y)]
\]

This objective does not enforce invariance across environments and allows the model to exploit correlations that do not generalize.

This project studies models that instead aim to learn representations \( Z = \phi(X) \) such that:

- Predictive performance is stable across environments
- Learned features are invariant to environment-specific noise

---

## Methodology

### 1. Synthetic Data with Multiple Environments
We generate controlled datasets where:
- The causal mechanism remains fixed
- Spurious correlations vary across environments

This allows precise analysis of model behavior under distribution shift.

### 2. Baseline Models (ERM)
Standard models (e.g., linear models, MLPs) trained with ERM serve as baselines to demonstrate failure modes.

### 3. Invariant Learning Models
We implement IRM-inspired models that optimize a composite loss:

\[
L = L_{\text{prediction}} + \lambda \cdot L_{\text{invariance}}
\]

The invariance penalty encourages representations that remain predictive across environments.

### 4. Out-of-Distribution Evaluation
Models are evaluated on unseen environments using:
- In-distribution vs OOD accuracy
- Generalization gap
- Stability metrics across environments





# learning-causality-not-correlation
This project demonstrates why traditional ML models fail in production when data distributions shift, and how causal-inspired methods like Invariant Risk Minimization (IRM) can build more robust models.


### Key Features:
- **Synthetic Data Generator**: Create customizable environments with causal and spurious variables
- **Multiple Baselines**: Traditional ML models (scikit-learn) and neural networks
- **Invariant Learning**: Implementation of IRM and variants in PyTorch
- **Comprehensive Evaluation**: Metrics for OOD generalization, robustness, and stability


🧪Experiments
Experiment	Description	Key Metric
Baseline ML	Traditional models (RF, XGBoost, MLP)	Accuracy drop OOD
Vanilla NN	Standard neural network	Generalization gap
IRM	Invariant Risk Minimization	Performance stability
Ablation	Component importance	Contribution analysis


<img width="1373" height="1582" alt="deepseek_mermaid_20260102_002d1a" src="https://github.com/user-attachments/assets/bc6f1eb7-1cbe-43f5-a532-312b0fd252e2" />





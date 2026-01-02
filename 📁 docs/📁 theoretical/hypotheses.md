

Markdown
# Formalized Hypotheses

## Main Hypothesis (H1)

**Invariant learning models (IRM) generalize better out-of-distribution than traditional ERM models.**

### Formalization
H1: gap_IRM < gap_ERM
where: gap = accuracy(E_test) − mean(accuracy(E₁), accuracy(E₂))


This hypothesis captures the core claim that enforcing invariance across environments improves robustness under distribution shift.

---

## Sub-Hypotheses

### H1.1: Learning of Spurious Correlations

ERM models assign significant importance to spurious variables (`s1`, `s2`), while IRM models suppress them.

**Metric**
importance_spurious_ERM > importance_spurious_IRM


**Statistical Test**
- Two-sample t-test on SHAP values for `s1` and `s2`

---

### H1.2: Identification of Causal Variables

IRM models more effectively identify causal variables (`x1`, `x2`) compared to ERM models.

**Metric**
importance_causal_IRM > importance_causal_ERM


**Statistical Test**
- Two-sample t-test on SHAP values for `x1` and `x2`

---

### H1.3: Representation Stability Across Environments

Latent representations learned by IRM models are more similar across environments than those learned by ERM models.

**Metric**
distance_IRM < distance_ERM


**Test**
- Cosine distance between embeddings extracted from different environments

---

## Null Hypothesis (H0)

There is no statistically significant difference in OOD generalization between ERM and IRM models.
H0: gap_IRM = gap_ERM


---

## Significance Level

All statistical tests are conducted at a significance level of:
α = 0.05


---

## Statistical Power

With a total sample size of **n = 10,000** and an assumed effect size of **Cohen’s d = 0.3**, the statistical power of the tests is estimated to be:
Power > 0.95


This ensures a high probability of detecting meaningful differences between ERM and IRM models if they exist.

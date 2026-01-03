 Causal Inference Primer for Machine Learning

## 1. From Correlation to Causation

### The Fundamental Difference
```

Correlation: P(Y | X)
Causation: P(Y | do(X))

```

**Key Insight**: Causal relationships are **stable under intervention**, while correlations can change with context.

### Simpson's Paradox Example
```

Hospital treatment example:

Treatment A success rate: 75% (young), 10% (elderly)
Treatment B success rate: 90% (young), 5% (elderly)

Aggregate results might show Treatment B better overall,
but actually Treatment A is better for both age groups.

```

## 2. The Ladder of Causation (Pearl, 2009)

### Level 1: Association (Seeing)
- **Questions**: "What is?" "How are variables related?"
- **Tools**: Regression, correlation, ML models
- **Limitation**: Cannot answer "what if" questions

### Level 2: Intervention (Doing)
- **Questions**: "What would happen if we do X?"
- **Tools**: Randomized controlled trials, do-calculus
- **Notation**: P(Y | do(X))

### Level 3: Counterfactuals (Imagining)
- **Questions**: "What would have happened if we had done X instead?"
- **Tools**: Structural causal models, counterfactual reasoning

## 3. Structural Causal Models (SCMs)

### Definition
An SCM is a tuple (U, V, F, P) where:
- U: Exogenous variables (external factors)
- V: Endogenous variables (internal to system)
- F: Structural functions determining V from U and other V
- P: Probability distribution over U

### Example: Loan Default SCM
```

U = {U_income, U_credit, U_error}
V = {income, credit_history, application_channel, loan_default}

Structural equations:

1. income = f₁(U_income)
2. credit_history = f₂(U_credit)
3. loan_default = I(0.5income + 0.3credit_history + U_error > threshold)
4. application_channel = g(loan_default, U)  # Spurious correlation

```

## 4. The do-operator and Interventions

### Mathematical Definition
```

P(Y | do(X = x)) = P_m(Y | X = x)

```
where P_m is the probability in the modified model where equation for X is replaced by X = x.

### Backdoor Criterion
To estimate causal effect of X on Y, we need to adjust for confounding variables Z:
```

P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z)

```
when Z satisfies:
1. Z blocks all backdoor paths from X to Y
2. Z does not contain descendants of X

## 5. Causal Graphs and d-separation

### Building Blocks
1. **Chain**: X → Z → Y
   - X and Y are dependent, but conditionally independent given Z
   
2. **Fork**: X ← Z → Y
   - X and Y are dependent, but conditionally independent given Z
   
3. **Collider**: X → Z ← Y
   - X and Y are independent, but conditionally dependent given Z

### d-separation Rule
Two sets of nodes A and B are d-separated by Z if all paths between them are blocked.

## 6. Invariant Causal Prediction (ICP)

### Core Principle
```

If X_j is a direct cause of Y, then the conditional distribution
P(Y | X_j, X_{-j}) should be invariant across environments.

```

### Algorithm Sketch
1. For each subset S of variables:
2. Test if P(Y | X_S) is invariant across environments
3. The intersection of all invariant subsets gives causal parents

## 7. Practical Causal Discovery Methods

### Constraint-Based Methods
- **PC Algorithm**: Uses conditional independence tests
- **FCI Algorithm**: Handles latent confounders

### Score-Based Methods
- **GES (Greedy Equivalence Search)**: Searches over DAG space
- **LiNGAM**: Linear non-Gaussian models

### Functional Causal Models
- **ANM (Additive Noise Models)**: Y = f(X) + N, N ⟂ X

## 8. Causal Machine Learning Applications

### 1. Causal Feature Selection
Select features based on causal relevance, not just predictive power.

### 2. Policy Learning
Learn policies that work across different environments.

### 3. Transfer Learning
Transfer knowledge when causal mechanisms are invariant.

### 4. Fairness
Ensure decisions are based on causal factors, not proxies.

## 9. Common Pitfalls in Causal Inference

### 1. Confounding Bias
Not adjusting for common causes of treatment and outcome.

### 2. Selection Bias
Conditioning on colliders or their descendants.

### 3. M-Bias
Conditioning on a pre-treatment variable that is a collider.

### 4. Overcontrol
Adjusting for mediators between treatment and outcome.

## 10. Causal Inference vs. Machine Learning

| Aspect | Traditional ML | Causal Inference |
|--------|---------------|------------------|
| Goal | Predict Y given X | Understand effect of interventions |
| Data | i.i.d. from single distribution | Multiple environments/interventions |
| Evaluation | Predictive accuracy | Causal effect estimation |
| Assumptions | Stationarity | Causal graph, no unmeasured confounding |

## 11. Tools and Libraries

### Python Libraries
1. **DoWhy**: End-to-end causal inference
2. **CausalML**: Causal machine learning algorithms
3. **pgmpy**: Probabilistic graphical models
4. **EconML**: Causal inference with ML

### R Libraries
1. **DAGitty**: Draw and analyze causal DAGs
2. **pcalg**: Causal structure learning

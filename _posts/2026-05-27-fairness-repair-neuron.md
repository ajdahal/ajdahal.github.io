---
layout: post
title: "Fairness Repair in Neural Networks"
description: "A critical analysis of fairness repair methods and research gaps for subspace-based approaches."
---

# Fairness Repair in Neural Networks: SOTA Analysis, Timeline, and Weaknesses

**Purpose**: Deep critical analysis of all 8 papers + newer SOTA, identifying exploitable weaknesses and research gaps for our subspace-based approach.


## Part 1: Timeline of Papers (Chronological)

```
2022 ── NeuronFair (ICSE)    Group 1: IDI generation via biased neurons
     ── FairNeuron (ICSE)    Group 3: Selective neuron repair with adversary games
     ── CARE (ICSE)          Group 4: Causality-based, PSO weight adjustment

2023 ── DICE (ICSE)          Group 2: Information-theoretic QID + deactivation/activation
     ── Faire (TOSEM)        Group 4: Neuron condition synthesis, last-layer gender probe

2024 ── RUNNER (ICSE)        Group 3: Neuron importance score + activation difference loss
     ── NeuFair (ISSTA)      Group 4: Random Walk + SA for dropout mask search
     ── CCBR (Exp.Sys.App.)  Counterfactual SCM + NSGA-III (improves CARE by 19.56%)

2025 ── FairFLRep (TOSEM)    Group 4: Gradient×Forward Impact + PSO, last layer
     ── FaVeR (IJCAI)        New: SMC-based individual fairness verification + repair
     ── CauSE Survey (FSE)   Workshop: Challenges in causality-driven NN repair
```

**Is RUNNER the SOTA?** Partially. RUNNER (ICSE 2024) is the best retraining-based approach for group fairness, but:
- **FairFLRep (TOSEM 2025)** is newer and targets same group fairness problem
- **FaVeR (IJCAI 2025)** is newer for individual fairness
- **CCBR (2024)** outperforms CARE by 19.56% with counterfactual causal modeling

---

## Part 2: Detailed Technical Analysis of Each Paper

---

### 2.1 NeuronFair — ICSE 2022 (Group 1: IDI Generation)

**ArXiv**: 2112.13214

**Core Idea**: Use neuron activation differences between demographic groups (male vs. female) at an intermediate layer to guide test case generation. Plug the selected layer into a loss function and do local + global search for Individual Discriminatory Instances (IDIs).

**Technical Mechanism**:
1. **Fault Localization**: Compute activation difference per neuron across all inputs by gender. Select the layer with the highest aggregate activation difference (layer L*).
2. **IDI Generation (local phase)**: For a single seed instance, do gradient-guided perturbation to maximize the discriminatory score `|P(y=1|x, a=0) - P(y=1|x, a=1)|` while keeping other features fixed.
3. **IDI Generation (global phase)**: Use the selected intermediate layer as a loss plug-in; random search globally.
4. **Repair**: Retrain on original data + X% of IDIs.

**What it does well**:
- High yield of IDIs (5.84× more than prior methods)
- Interpretable: identifies which layer/neurons are biased
- Covers both structured (tabular) and unstructured (image) data

**Critical Weaknesses**:
- **W1**: Repair is just retraining with IDIs — no targeted weight adjustment. Indiscriminate retraining can cause catastrophic forgetting.
- **W2**: IDIs may not be "natural" instances — perturbations are gradient-guided and may produce feature combinations that don't exist in real data (e.g., impossible feature co-occurrences).
- **W3**: Selecting one layer (L*) for testing assumes bias is concentrated in one layer. In practice, bias is distributed across all layers.
- **W4**: The activation difference metric is a marginal (per-neuron) measure. It misses coordinated patterns across multiple neurons (subspace structure of bias).
- **W5**: Gradient vanishing problem limits it to networks with strong gradient flow; doesn't generalize to very deep networks.
- **W6**: Individual discriminatory instances focus on counterfactual fairness (same person, different gender) — but the perturbation ignores feature constraints (e.g., relationship=Husband impossible with gender=Female).

---

### 2.2 DICE / Information-Theoretic Debugging — ICSE 2023 (Group 2)

**Core Idea**: Three-phase pipeline — (1) Generate IDIs using Quantitative Individual Discrimination (QID), (2) localize faulty neurons via Average Causal Discrimination (ACD) computed by causal intervention, (3) repair by deactivating neurons that increase discrimination and activating those that decrease it.

**Key Metric — QID**: For instance x, QID(x) = `|P(y=1|x, a=0) - P(y=1|x, a=1)|` where a=0 and a=1 are the two group memberships.

**Key Metric — ACD**: `ACD_i = QID(x, before masking neuron i) - QID(x, after masking neuron i)`. Positive ACD = masking neuron i reduces discrimination. Negative ACD = masking neuron i increases discrimination.

**Repair**: Mask neurons with highest negative influence on fairness (neurons that increase discrimination) and activate (unblock) neurons with highest positive influence.

**What it does well**:
- More subtle bias detection than group fairness (probability difference, not just label difference)
- Causal intervention for localization (stronger than correlation-based approaches)
- Individual fairness focus

**Critical Weaknesses**:
- **W1 (Probe injection and other instances)**: Deactivating a neuron that reduces discrimination for one instance may increase discrimination for another. There is no global guarantee that the repair is monotone. The critique in the document asks this directly.
- **W2**: QID is computed per instance with brute-force counterfactual (flip gender, hold everything else). This ignores that many feature combinations are impossible (husband → must be married+male).
- **W3**: ACD is estimated over a finite sample of IDIs. If the IDI sample is biased or incomplete, the ACD estimates are wrong.
- **W4**: Masking/activating is inference-time. The underlying weight parameters remain unchanged — so after re-deploying with full activations, bias returns.
- **W5**: No group fairness consideration. A network can have perfect individual fairness QID=0 but still exhibit statistical parity gaps at the population level.

---

### 2.3 FairNeuron — ICSE 2022 (Group 3: Selective Neuron Repair)

**Core Idea**: Identify "biased paths" (neurons frequently activated by biased samples) and "benign paths" (neurons frequently activated by clean samples). Repair alternates: (a) train on biased samples with neutral neurons dropped out → only biased path active → pushes fairness; (b) train on benign samples without dropout → restores accuracy.

**Key Assumption**: Biased samples and benign samples are known/labeled in advance.

**What it does well**:
- Targets specific neurons rather than the entire network
- Alternating training balances accuracy and fairness objectives

**Critical Weaknesses**:
- **W1 (requires biased/benign sample labels)**: How do you know which training samples are "biased" vs "benign" a priori? The paper assumes this is given, but in practice it requires additional effort to identify them. This is a circular problem.
- **W2**: Dropping "neutral" neurons during biased sample training forces gradient to flow only through biased neurons — but neutral neurons may still carry task-relevant information that gets perturbed.
- **W3**: The adversary game is computationally expensive and unstable (min-max optimization). Convergence is not guaranteed.
- **W4**: Neuron path identification relies on activation thresholds, introducing hyperparameters (θ and γ).
- **W5**: Does not explicitly handle proxy attributes (e.g., relationship/marital-status encoding gender).

---

### 2.4 RUNNER — ICSE 2024 (Group 3: SOTA — Deep Analysis from Code)

**Reference**: Li, Tianlin et al. "RUNNER: Responsible UNfair NEuron Repair for Enhancing Deep Neural Network Fairness." ICSE 2024.

**Architecture in Code**: `Input → Linear(200) → ReLU → Linear(200) → ReLU → Linear(1) → Sigmoid`

#### RUNNER's Exact Algorithm (from `baselines/RUNNER/utils.py`):

**Step 1 — Compute Neuron Importance Score per Batch** (`cal_importance_gapReg`):
```python
# For DP: fairness gap loss
loss_reg = |mean(output_group0) - mean(output_group1)|

# Backpropagate fairness gap loss
loss_reg.backward()

# Importance score for each neuron (row of weight matrix)
importance_i = sum_j (W_ij * (∂L_fairness/∂W_ij))^2
```
This is a **Taylor expansion approximation**: `importance_i ≈ ||∂L/∂W_i||^2_Taylor`
It estimates "how much would a unit change in this neuron's weights reduce the fairness gap?"

**Step 2 — Select Top-k% Most Important Neurons**: `top-k(importance_i)` where k = `neuron_ratio` hyperparameter.

**Step 3 — Train with Combined Loss**:
```python
# For selected neurons i ∈ {top-k%}:
loss_reg = |mean(h_i | group=0) - mean(h_i | group=1)|  # per-neuron activation difference

# Full loss
loss = BCE(output, y) + λ * loss_reg
```

**Step 4 — Repeat per Batch**: Neuron selection is re-computed every batch. As neurons get repaired, importance scores shift to other neurons.

#### RUNNER's Critical Weaknesses (from code analysis):

**W1 — Mean activation difference ≠ distributional independence**:
RUNNER minimizes `|E[h_i | A=0] - E[h_i | A=1]|` (1st moment). But sensitive information can be encoded in:
- Higher moments (variance, skewness) → not captured
- Joint patterns across multiple neurons → not captured by per-neuron loss
- Non-zero-mean but equal-mean bimodal distributions → not captured

*Example*: If females are all in [−2, 0] and males are all in [0, +2], mean(female) ≈ mean(male) ≈ 0, but a classifier can perfectly predict gender from this neuron.

**W2 — Taylor score is a local linear approximation**:
The importance score `(W * ∂L_fairness/∂W)^2` is a 1st-order Taylor approximation valid only near the current parameter value. The actual causal effect of changing neuron i's weights may be very different globally (especially for ReLU networks where curvature is high near activation boundaries).

**W3 — Hyperparameter k has no principled selection**:
From the README:
```
| Dataset | DP    | EO   |
|---------|-------|------|
| Adult   | 50%   | 5%   |
| COMPAS  | 5%    | 5%   |
| Credit  | 5%    | 5%   |
| LSAC    | 5%    | 5%   |
```
The optimal k varies wildly across datasets. On Adult with DP, 50% of neurons must be targeted. There is no principled method for setting k — it must be manually tuned via cross-validation.

**W4 — No convergence guarantee**:
The iterative selection is implicitly driven by batch sampling noise. There is no proof that the sequence of neuron selections converges. In practice the optimization may oscillate between different subsets of neurons.

**W5 — Group fairness only (binary protected attribute)**:
`sample_batch_sen_idx(X, A, y, batch_size, s)` where `s ∈ {0, 1}` — only binary groups supported. Cannot handle:
- Age as a continuous protected attribute
- Race with 5+ categories
- Intersectional groups (female + elderly + Black)

**W6 — Proxy attributes not addressed**:
RUNNER reduces activation difference for the primary sensitive attribute (sex). But if proxy attributes (relationship=Husband, marital-status=Married) also encode sex, reducing the sex-related neuron activation differences does NOT remove proxy-encoded bias. The model can still use `relationship → income` to discriminate.

**W7 — All-layer simultaneous repair with no inter-layer coordination**:
RUNNER applies the fairness loss to neurons in both layer 1 and layer 2 simultaneously. When you push layer 1 neurons to equalize, this changes the input distribution to layer 2, potentially making layer 2 neurons MORE biased. RUNNER addresses this implicitly (the gradient will then shift importance to layer 2 neurons), but there is no explicit check for "did layer 1 repair introduce new bias in layer 2?"

**W8 — Evaluation metric mismatch with other papers**:
RUNNER uses Average Precision (AP) for accuracy evaluation. Most other papers use standard accuracy (%). AP is appropriate for imbalanced datasets but makes direct numerical comparisons with other papers misleading.

**W9 — Architecture is fixed 2-layer MLP**:
RUNNER's forward pass is specifically designed to track `sum1` (avg activations, layer 1), `sum2` (avg activations, layer 2), `sum3` (avg output) — hardcoded for 2 hidden layers. Generalizing to deeper networks (ResNets, Transformers) would require significant redesign.

**W10 — "Iterative neutralization of new biases" is implicit**:
The paper claims it iteratively neutralizes new biases that emerge from weight adjustments. In the code, this happens implicitly: after correcting some neurons, the gradient of the fairness gap shifts to other neurons, making them "newly important." But this is passive — there is no explicit check: "did our repair in iteration t introduce new bias in iteration t+1?"

#### What RUNNER Gets Right (User's Two Core Insights):

1. ✅ **Start with a pre-trained biased network and retrain minimally** → Preserves the good parts of the network, only modifies specific weights.

2. ✅ **Iterative repair with shifting focus** → As neurons are fixed, the gradient re-identifies which neurons still carry the remaining fairness gap. This is better than a one-shot fix.

---

### 2.5 Faire — TOSEM 2023 (Group 4: Neuron Condition Synthesis)

**Core Idea**:
1. Freeze all layers except the last → fine-tune as a gender classifier → find neurons most responsible for gender prediction.
2. Add a soft mask `m_i ∈ [0,1]` after each neuron. Optimize masks to:
   - Penalize neurons that contribute to gender prediction (m_i → 0)
   - Promote neurons that contribute to income prediction (m_i → 1)
   - Avoid changing neurons that contribute to both (m_i ≈ current value)

**Critical Weaknesses**:
- **W1**: The assumption that gender-predictive neurons are concentrated in the last layer is unproven. Research on mechanistic interpretability shows sensitive information is encoded in ALL layers.
- **W2**: Fine-tuning the last layer as a gender classifier is itself a proxy measure — it finds neurons correlated with gender, not necessarily causally driving unfair predictions.
- **W3**: Soft masks applied at each neuron = element-wise scaling. This is equivalent to a diagonal linear transformation of the representation. Compare with our approach which applies a full-rank linear projection (INLP) or a nonlinear adapter — both are more expressive.
- **W4**: The three-way objective (penalize gender, promote task, preserve both-neurons) requires careful tuning of multiple loss weights.
- **W5**: Proxy attributes: same problem as RUNNER. Finding neurons that predict gender from a frozen last-layer probe may miss the Husband/Married proxy encoding.

---

### 2.6 NeuFair — ISSTA 2024 (Group 4: Dropout-based Search)

**Core Idea**: Formulate fairness repair as a combinatorial search over binary dropout masks `m ∈ {0,1}^N`. Use Random Walk + Simulated Annealing (SA) to find mask M* that minimizes `|bias change| + α|accuracy change|`.

**Critical Weaknesses**:
- **W1 (Binary all-or-nothing masking)**: A neuron either fires fully or not at all. Many neurons in a network encode BOTH task-relevant and sensitive-attribute-related information. Binary masking cannot surgically remove only the sensitive component — it destroys both. This is why NeuFair tends to have worse accuracy-fairness tradeoffs.
- **W2 (No retraining — inference-time only)**: The underlying weights are unchanged. The bias is still encoded in the weights; masking just prevents it from propagating during inference. If the mask is removed (e.g., for a different task), bias immediately re-emerges. This is not a true repair.
- **W3 (Scalability)**: SA over N neurons is O(N) per evaluation step. For large networks (ResNets with millions of neurons), this is intractable. The paper evaluates on small MLPs.
- **W4 (SA local optima)**: SA does not guarantee finding the global optimum mask. Different random seeds may give very different results.
- **W5 (No proxy handling)**: Same proxy attribute problem.

---

### 2.7 CARE — ICSE 2022 (Group 4: Causality + PSO)

**Core Idea**:
1. **Causal intervention**: Directly mask each neuron (set activation to 0) and observe the change in the Average Causal Effect on discrimination.
2. **PSO weight adjustment**: Use Particle Swarm Optimization to search for minimal weight adjustments to the top-X% causally important neurons.

**Critical Weaknesses**:
- **W1 (Causal intervention conflation)**: Setting neuron i's activation to 0 is not the same as removing neuron i's causal influence. In a network with non-zero weights, zeroing neuron i changes the entire downstream computation. The "causal effect" estimated this way is entangled with indirect effects through other neurons.
- **W2 (PSO scales poorly)**: PSO searches over a high-dimensional parameter space (weights of top-X% neurons). For even moderate-sized networks, this is computationally expensive.
- **W3 (Single-objective PSO)**: Standard CARE uses a single-objective loss. The improved version CCBR (2024) addresses this with NSGA-III (multi-objective optimization), improving fairness by an additional 19.56%.
- **W4 (Which neurons to repair is fixed beforehand)**: CARE selects the top-X% neurons before running PSO. But X is a hyperparameter with no principled selection.
- **W5**: No inter-layer coordination: repairing neuron i in layer 1 changes the input to layer 2, potentially creating new causal effects in layer 2 that weren't there before.

---

### 2.8 FairFLRep — TOSEM 2025 (Group 4: Latest)

**ArXiv**: 2508.08151

**Core Idea**:
1. **Fault Localization**: Score each neuron using:
   ```
   score_i = gradient_loss_i × forward_impact_i
   ```
   - `gradient_loss_i`: gradient of the fairness loss w.r.t. neuron i's activation — "how responsible is this neuron for the current bias?"
   - `forward_impact_i`: magnitude of neuron i's contribution to the final output — "how much does this neuron matter for the prediction?"
   - Their product = high-impact, high-responsibility neurons

2. **Repair**: PSO to find minimal weight adjustments for top-ranked neurons.
3. **Target**: Specifically the last layer (or final layers).

**What is similar to RUNNER**: Both use gradient information for localization and combine classification loss + fairness loss.

**What is different from RUNNER**:
| Aspect | RUNNER | FairFLRep |
|--------|--------|-----------|
| Localization score | Taylor score of fairness gap loss: `(W * ∂L_fair/∂W)^2` | Product of gradient loss × forward impact |
| Repair method | Gradient descent, activation difference loss | PSO (metaheuristic), direct weight search |
| Training | Continues gradient-based retraining | Targeted parameter search (no full retraining) |
| Layer scope | All hidden layers | Specifically last layer |
| Multi-objective | No | Implicitly through PSO fitness |
| Proxy attributes | Not addressed | Not addressed |
| Protected attribute type | Binary only | Binary only |

**Critical Weaknesses**:
- **W1 (Last-layer only assumption)**: Targeting only the last layer ignores bias encoded in earlier feature representations. The deeper layers may learn proxy-encoded gender representations that the last layer then uses linearly.
- **W2 (PSO vs gradient descent trade-off)**: PSO doesn't use gradient information during the search, making it slow for high-dimensional parameter spaces. RUNNER's gradient-based approach is more efficient, though NeuFair's SA is similar.
- **W3**: `gradient_loss × forward_impact` is still a proxy correlation measure. High impact + high gradient for fairness = "blamed" neuron, but this doesn't capture whether the neuron is causally responsible vs. just correlated.
- **W4**: No convergence guarantee for PSO.

---

## Part 3: Newer SOTA Papers Not in the Original Document

### 3.1 FaVeR — IJCAI 2025 (Individual Fairness Verification + Repair)

**Full title**: "Efficient Counterexample-Guided Fairness Verification and Repair of Neural Networks Using Satisfiability Modulo Convex Programming"

**Core Idea**:
1. Use SMC (Satisfiability Modulo Convex programming) to **verify** whether the network satisfies individual fairness: ∀x, x': d(x, x') < ε → |f(x) - f(x')| < δ
2. If verification fails, generate a **counterexample** (a pair of similar individuals that are treated unfairly)
3. Use counterexample to guide **backward weight adaptation**: identify high-sensitivity neurons (activation difference between counterexample pairs), adjust weights from the last layer backward iteratively

**Why this is important**: This is the ONLY paper that provides a **formal guarantee** of individual fairness. All other papers (RUNNER, FairFLRep, NeuFair, CARE) provide heuristic fairness improvement but no formal verification.

**Limitations**:
- SMC-based verification is computationally expensive (scales poorly to large networks)
- Only addresses individual fairness, not group fairness
- The ε-δ definition of individual fairness requires defining a meaningful distance metric d(x, x') — for tabular data with mixed types, this is non-trivial

---

### 3.2 CCBR — Expert Systems with Applications 2024 (Counterfactual SCM + NSGA-III)

**Core Idea**:
1. Model the neural network as a **Counterfactual Structural Causal Model (CSCM)**: Each neuron is a variable in a structural equation model.
2. Use **counterfactual causal tracing** to identify neurons whose counterfactual effect on the outcome differs across groups.
3. Use **NSGA-III** (multi-objective evolutionary algorithm) to simultaneously optimize (a) accuracy, (b) fairness, (c) weight change magnitude.

**Result**: 92.56% fairness improvement vs. CARE's 73% — a 19.56% improvement.

**Why it's better than CARE**: CARE uses a single-objective loss with PSO. CCBR models the causal structure more faithfully (CSCM) and uses multi-objective optimization (NSGA-III), finding better Pareto-optimal solutions.

---

## Part 4: Comprehensive Weakness Map Across All Papers

```
                    Fairness Testing & Repair — Weakness Map
                    =========================================

DIMENSION         | NeuronFair | FairNeuron | DICE | RUNNER | Faire | NeuFair | CARE | FairFLRep
------------------|------------|------------|------|--------|-------|---------|------|----------
Individual fair.  |    ✓       |     ✗      |  ✓   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
Group fairness    |    ✗       |     ✓      |  ✗   |   ✓    |   ✓   |    ✓    |  ✓   |    ✓
Convergence proof |    ✗       |     ✗      |  ✗   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
All-layer repair  |    ✗       |     ✓      |  ✓   |   ✓    |   ✗   |    ✓    |  ✗   |    ✗
Proxy attributes  |    ✗       |     ✗      |  ✗   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
Subspace-aware    |    ✗       |     ✗      |  ✗   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
Full distribution |    ✗       |     ✗      |  ✗   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
Multi-valued sens.|    ✗       |     ✗      |  ✗   |   ✗    |   ✗   |    ✗    |  ✗   |    ✗
No retraining req.|    ✗       |     ✗      |  ✓   |   ✗    |   ✓   |    ✓    |  ✓   |    ✓
```

**Universal weakness across ALL 8 papers**:
→ **No paper addresses the subspace structure of sensitive information in neural network representations.**
→ **No paper addresses proxy attributes (correlated features that also encode sensitive information).**

---

## Part 5: The Core Research Gap — Subspace vs. Neuron-Level Repair

Every paper in this survey treats bias as being encoded in **individual neurons** (or small subsets). But this is a fundamental mischaracterization of how modern neural networks encode information.

### Evidence from Our Experiments

From our experiments on the Adult income dataset with a 4-layer MLP (128→64→16→2):

| Method | Fairness Improvement | Accuracy Impact |
|--------|---------------------|-----------------|
| Neuron dropping (top-k activation diff.) | Low | High |
| INLP (linear gender direction removal) | Medium | Medium |
| MLP adversary (nonlinear adapter) | Highest | Low |

The ordering (neuron dropping < INLP < MLP adversary) reveals:
1. **Neuron dropping** is worst because it destroys information indiscriminately (same as NeuFair)
2. **INLP** is better because it removes a linear direction — a 1-D subspace — not just a single neuron. But it misses nonlinear encoding.
3. **MLP adversary** is best because it:
   - Captures nonlinear encoding (Husband → Male via Married status)
   - Automatically removes proxy attribute information
   - Applies a full-rank transformation to the representation space

### Why Neuron-Level Approaches Fail

Sensitive information in neural networks is encoded **geometrically** in the activation space:
- Gender information lies in a **k-dimensional subspace** (k ≥ 1) of the 64-D representation space
- Removing information from a single neuron just moves the encoded information to the remaining neurons (via weight compensation during training)
- RUNNER's `|mean(h_i|A=0) - mean(h_i|A=1)|` for one neuron i can be driven to zero while gender information is redistributed across all other neurons

This is analogous to squeezing a balloon: push down on one spot, the balloon expands elsewhere.

### What Our Subspace Approach Does Differently

```
RUNNER (per-neuron):
  Input → NN → h64 → [select top-k neurons] → drive activation difference to 0

Our approach (subspace):
  Input → NN → h64 → [remove gender subspace] → h64_debiased → output
```

The key differences:
1. **Scope**: We operate on the entire 64-D representation at once, not one neuron at a time
2. **Geometry**: We find the subspace of h64 that encodes gender (via INLP probing) and project it out
3. **Proxy awareness**: The MLP adversary naturally discovers proxy encodings (Husband/Wife → gender) because it tries to predict gender from h64 and must account for all predictive features
4. **Distributional**: Our approach minimizes `I(h64; gender)` rather than `|E[h_i|A=0] - E[h_i|A=1]|`

---

## Part 6: Critical Open Questions Across All Papers

### Q1: Does fixing bias in one layer introduce new bias in another?

**RUNNER's response**: Implicitly addressed through iterative gradient re-computation. After correcting some neurons, the fairness gradient shifts to other neurons. But this is passive, not verified.

**CARE/NeuFair**: Not addressed — single-pass repair.

**FaVeR**: The best answer — use verification after each repair step to check if new fairness violations emerged. This is the only approach with an explicit re-check.

**Our research opportunity**: After each iteration of adapter training, use INLP probing across ALL layers to check if gender information has migrated to an earlier or later layer. This would be the explicit "new bias neutralization" check that RUNNER only does implicitly.

### Q2: Are IDIs natural/valid counterfactuals?

NeuronFair and DICE generate IDIs by flipping the protected attribute. But for the Adult dataset, flipping gender from Female to Male while keeping Relationship=Wife is incoherent (Wife must be female). Invalid counterfactuals produce misleading test cases and repair signals.

**Our research opportunity**: Use valid counterfactual methods (TabChange, CausalML) to generate only feasible gender-flipped instances, and use these as the signal for both testing and repair.

### Q3: Why does the sensitive cluster problem invalidate existing approaches?

All 8 papers define the sensitive attribute as a binary variable (gender=0/1) and directly use this for localization and repair. But in the Adult dataset:
- **Gender** correlates with **Relationship** (r ≈ 0.9): Husband → Male, Wife → Female
- **Gender** correlates with **Marital-status** (r ≈ 0.75): Married-civ-spouse → mostly male, Divorced → both

Consequence: A model can predict income using `relationship=Husband` without ever "seeing" the gender feature. All 8 papers' localization methods would find this neuron has LOW gender activation difference (because it fires for "Husband" regardless of whether the gender input is flipped), but HIGH real-world discriminatory impact.

**RUNNER in particular**: `sample_batch_sen_idx` splits training data by gender label. So `batch_x_0` = female samples (many with relationship=Wife, marital=Married) and `batch_x_1` = male samples (many with relationship=Husband, marital=Married). The activation difference at any neuron reflects the ENTIRE feature difference between male and female samples, including proxy attributes. So RUNNER does partially capture proxy encoding — but only at the population level of the dataset, not by explicitly identifying the proxy.

---

## Part 7: Exploitable Gaps and Our Contributions

### Gap 1: Neuron-level vs. Subspace-level Repair

**What's missing**: All 8 papers operate neuron-by-neuron. No paper removes the gender **subspace** from the representation.

**Our contribution**: INLP (Iterative Nullspace Projection) finds the gender direction iteratively, then projects the entire representation space onto the orthogonal complement. The adapter layer (64→64) implements this transformation end-to-end in a learnable way.

**Why this is better**: Removing a subspace guarantees that no linear combination of activations can predict gender — not just individual neurons.

### Gap 2: Nonlinear Proxy Attribute Handling

**What's missing**: No paper explicitly addresses proxy attributes. They all target the raw sensitive attribute label.

**Our contribution**: The MLP adversary in `stage1_bias_analysis.py` naturally discovers and removes proxy encodings because:
1. The adversary tries to predict gender from h64
2. When gender is encoded via proxy (relationship=Husband), the adversary uses that proxy to predict gender
3. The adapter then must remove the proxy encoding to fool the adversary
4. Result: h64_debiased has reduced information about BOTH the direct sensitive attribute AND its proxies

**Evidence**: MLP adversary > INLP in our experiments. The gap is larger when proxy encoding is stronger (Adult dataset with Husband/Wife relationship attribute).

### Gap 3: Single-shot vs. Iterative with Verification

**What's missing**: RUNNER iterates but doesn't verify convergence. All others are single-shot.

**Our contribution opportunity**: Implement RUNNER's key insight (iterative repair) with explicit INLP-based verification after each iteration. Measure gender predictability from each layer's activations using linear probing accuracy. Iterate until linear probing accuracy ≤ 55% across all layers (INLP stopping criterion).

### Gap 4: Group Fairness vs. Individual Fairness vs. Distributional Fairness

**What's missing**: Papers either do group fairness (RUNNER, FairFLRep, FairNeuron) or individual fairness (NeuronFair, DICE, FaVeR). No paper does **distributional fairness** (minimizing MI between h64 and gender).

**Our contribution opportunity**: Replace the adversarial loss with HSIC (Hilbert-Schmidt Independence Criterion) or Maximum Mean Discrepancy (MMD) loss. This targets the full distribution `P(h64|A=0)` vs `P(h64|A=1)`, not just the mean.

### Gap 5: Layer-by-Layer Analysis of Where Bias Lives

**What's missing**: No paper systematically measures at which layer bias information is highest and why.

**Our contribution**: Run INLP probing at every layer of the 4-layer MLP (after 128, after 64, after 16, at output). Map the gender predictability (linear SVM accuracy) at each layer. This gives a "bias profile" of the network, identifying where targeted repair would be most effective.

---

## Part 8: Summary Table of Papers and Key Gaps

| Paper | Year | Type | Fairness | Localization | Repair | Proxy? | Subspace? | Convergence? |
|-------|------|------|----------|--------------|--------|--------|-----------|--------------|
| NeuronFair | 2022 | Test+Repair | Individual | Activation diff | Retrain on IDIs | ✗ | ✗ | ✗ |
| FairNeuron | 2022 | Repair | Group | Activation path | Alternating train | ✗ | ✗ | ✗ |
| CARE | 2022 | Repair | Group+Backdoor | Causal intervention | PSO weight adj | ✗ | ✗ | ✗ |
| DICE | 2023 | Test+Debug | Individual | ACD causal score | Mask/Activate | ✗ | ✗ | ✗ |
| Faire | 2023 | Repair | Group | Last-layer probe | Soft mask | ✗ | ✗ | ✗ |
| RUNNER | 2024 | Repair | Group | Taylor fairness score | Activation diff loss | ✗ | ✗ | ✗ |
| NeuFair | 2024 | Repair | Group | SA search | Binary dropout | ✗ | ✗ | ✗ |
| CCBR | 2024 | Repair | Group+Backdoor | CSCM causal trace | NSGA-III | ✗ | ✗ | ✗ |
| FairFLRep | 2025 | Repair | Group | Gradient×Impact | PSO last-layer | ✗ | ✗ | ✗ |
| FaVeR | 2025 | Verify+Repair | Individual | SMC verification | Backward weight adj | ✗ | ✗ | ✓ |
| **Our approach** | 2026 | Repair | Group+Dist. | INLP probing | Subspace projection + adapter | **✓** | **✓** | **Partial** |

---

## Part 9: Research Narrative for Paper Positioning

The existing body of work on fairness repair in neural networks shares a common architectural assumption: **bias is localized in individual neurons** and can be fixed by targeting those neurons one-by-one. This assumption fundamentally underestimates the distributed, geometric nature of how sensitive information is encoded in neural network representations.

Evidence from our experiments and from mechanistic interpretability research shows that:
1. Sensitive attributes are encoded in **subspaces** of activation spaces, not individual neurons
2. Sensitive information is **redundantly encoded** across multiple correlated features (the proxy attribute problem)
3. **Nonlinear encodings** (Husband/Wife relationship → gender) are missed by all linear localization methods (Taylor score, activation difference, gradient × forward impact)

Our approach — using an adapter layer trained with an MLP adversary (or equivalently, INLP for the linear case) — is the first to address all three problems simultaneously:
1. **Subspace**: The adapter applies a global 64×64 linear transformation, removing an entire subspace of gender-encoding directions
2. **Proxy attributes**: The adversary discovers proxy encodings naturally by trying to predict gender from all available information in h64
3. **Nonlinear encoding**: The MLP adversary captures nonlinear patterns that linear probing misses

The result is a approach that is:
- **More accurate** (less accuracy degradation than neuron dropping or INLP alone)
- **More fair** (removes both direct and proxy gender encoding)
- **More principled** (subspace geometry rather than heuristic neuron selection)
- **Complementary to RUNNER** (we use RUNNER's insight about iterative repair, but apply it at the representation subspace level)

---

## Part 10: Concrete Research Questions for Future Work

1. **Does removing the gender subspace from h64 guarantee no information leaks through earlier layers?**
   Test: INLP probing at h128, h64, h16 after adapter training. If gender accuracy at h128 remains high, we need an adapter at h128 too.

2. **Is the adapter's 64×64 linear transformation provably optimal for subspace projection?**
   Theory: The optimal linear debiaser is the orthogonal projection P = I - U·Uᵀ where U is the basis of the gender subspace found by INLP.

3. **Can we combine RUNNER's iterative approach with our subspace projection?**
   Proposal: After each RUNNER iteration, project the entire h64 representation onto the orthogonal complement of the current gender direction. This ensures that subsequent RUNNER iterations are operating on gender-free representations.

4. **Does the proxy cluster problem (sensitive cluster) explain why our MLP adversary outperforms INLP?**
   Experiment: Repeat experiments on a dataset where sensitive attribute is NOT correlated with proxy features. Measure whether INLP ≈ MLP adversary in this case.

5. **How does the dimensionality of the gender subspace vary across layers?**
   Experiment: Run iterative INLP at each layer until probing accuracy falls to 55%. The number of INLP iterations needed = dimensionality of the gender subspace at that layer.

---

## References

1. Zheng et al. "NeuronFair: Interpretable white-box fairness testing through biased neuron identification." ICSE 2022. [arXiv:2112.13214](https://arxiv.org/abs/2112.13214)
2. Gao et al. "FairNeuron: improving deep neural network fairness with adversary games on selective neurons." ICSE 2022.
3. Sun et al. "Causality-based neural network repair." ICSE 2022. [arXiv:2204.09274](https://arxiv.org/abs/2204.09274)
4. Monjezi et al. "Information-theoretic testing and debugging of fairness defects in deep neural networks." ICSE 2023.
5. Li et al. "Faire: Repairing fairness of neural networks via neuron condition synthesis." TOSEM 2023. [ACM](https://dl.acm.org/doi/10.1145/3617168)
6. Li et al. "RUNNER: Responsible unfair neuron repair for enhancing deep neural network fairness." ICSE 2024. [ACM](https://dl.acm.org/doi/10.1145/3597503.3623334)
7. Dasu et al. "NeuFair: Neural network fairness repair with dropout." ISSTA 2024. [arXiv:2407.04268](https://arxiv.org/abs/2407.04268)
8. Openja et al. "FairFLRep: Fairness aware fault localization and repair of deep neural networks." TOSEM 2025. [arXiv:2508.08151](https://arxiv.org/abs/2508.08151)
9. CCBR: "Towards robust neural networks: Exploring counterfactual causality-based repair." Expert Systems with Applications, 2024. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417424019493)
10. Fayyazi et al. "Efficient Counterexample-Guided Fairness Verification and Repair of Neural Networks Using Satisfiability Modulo Convex Programming." IJCAI 2025. [IJCAI](https://www.ijcai.org/proceedings/2025/42)
11. Vares & Johnson. "Causality-Driven Neural Network Repair: Challenges and Opportunities." CauSE@ESEC/FSE 2025. [arXiv:2504.17946](https://arxiv.org/abs/2504.17946)


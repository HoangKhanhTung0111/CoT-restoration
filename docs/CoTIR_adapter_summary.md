```
# Internalized Chain-of-Thought Reasoning (CoTIR) Summary
Source Paper: "Universal Image Restoration via Internalized Chain-of-Thought Reasoning" (CoTIR 2026)

## 1. Core Problem &amp; Concept
- **Internalized CoT Reasoning:** Rather than multi-pass tool invocation (which incurs high latency and error accumulation), CoTIR embeds structured reasoning ("Thinking -&gt; Planning -&gt; Action") directly into the feature representation in a single forward pass [12-14].
- **Three-Stage Reasoning Pipeline:**
  1. **Thinking (Feature Disentanglement):** Extracting $T_s$ (scene inherent features) and $T_d$ (degradation pattern identification) [13, 15].
  2. **Planning (Strategic Recovery Formulation):** Formulating $T_p$ (restoration plan analyzing interplay between $T_s$ and $T_d$) [13, 15].
  3. **Action (Guided Restoration):** Restorer recovers details guided by $T_{CoT}$ [13, 15].

## 2. CoT Adapter Architecture &amp; Feature Projection

### Projection to Semantic Subspaces
Given context features $\bar{T}^{(S)}$ processed after $S$ CoTIRBlocks [16, 17]:
$$\bar{T}_s = \text{Proj}_s(\bar{T}^{(S)})$$
$$\bar{T}_d = \text{Proj}_d(\bar{T}^{(S)})$$
$$\bar{T}_p = \text{Proj}_p(\text{Concat}(\bar{T}_s, \bar{T}_d))$$

### Zero-Initialized Gated Aggregation
Features are aggregated via zero-initialized learnable gates $(\sigma_s, \sigma_d, \sigma_p)$ to form CoT guidance [17]:
$$\bar{T}_{CoT} = \bar{T} + \sigma_s \bar{T}_s + \sigma_d \bar{T}_d + \sigma_p \bar{T}_p$$

## 3. Optimization via Learnable Lagrange Multipliers

### Multi-Constraint Loss Objective
$$\mathcal{L}(\theta, \lambda) = \mathcal{L}_{main}(\theta) + \sum_{i \in \{s, d, p\}} \lambda_i \cdot \text{sg}\left( \|\hat{c}_i - c_i\|_2^2 - \delta_i \right)$$
where:
- $c_i, \hat{c}_i$: Ground-truth vs predicted reasoning targets for scene ($s$), degradation ($d$), and plan ($p$) [18, 19].
- $\delta_i$: Allowed tolerance threshold per constraint [18, 19].
- $\text{sg}(\cdot)$: Stop-gradient operator for dual-optimizer stabilization [19, 20].
- $\lambda = (\lambda_s, \lambda_d, \lambda_p) \ge 0$: Learnable Lagrange multipliers updated dynamically via gradient ascent to adaptively balance restoration loss and reasoning constraints [19-21].

```
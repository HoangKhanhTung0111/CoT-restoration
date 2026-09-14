```
# Chain-of-Restoration (CoR) &amp; Universal Image Restoration (UIR) Summary
Source Paper: "Chain-of-Restoration: Multi-Task Image Restoration Models are Zero-Shot Step-by-Step Universal Image Restorers" (UIR 2024)

## 1. Core Problem: Degradation Coupling &amp; UIR Setting
- **Universal Image Restoration (UIR):** Models are trained on a set of degradation bases (e.g., noise, rain, haze) and tested on unknown composite degradations (combinations of bases) in a zero-shot manner [1, 2].
- **Degradation Coupling Issue:** When multiple degradations co-occur (e.g., low-light + rain + haze), attempting to remove all degradations in a single pass causes coupling distortion, where removing one degradation distorts features of another [3, 4].
- **Solution Concept:** Step-by-step restoration (Chain-of-Restoration). A multi-task restoration model removes one degradation basis per step guided by a Degradation Discriminator (DD) [5, 6].

## 2. Mathematical Formulation &amp; Logic

### Step-by-step Restoration Formulation
For an input image \\(X_0\\) with composite degradation comprising \\(T\\) components [7, 8]:
\[X_i = M(X_{i-1}, \text{type}_i), \quad i = 1, 2, \dots, T\]
where \\(\text{type}_i = \text{Degradation\_Discriminator}(X_{i-1})\\).

### Degradation Discriminator (DD) Selection Logic
For a non-blind multi-task model, DD outputs a probabilistic vector \\(v \in \mathbb{R}^{n+1}\\) over \\(n\\) degradation bases plus clean state [9, 10]. Soft margins prioritize higher-order bases and control sequence [10, 11]:
\[v'[idx] = v[idx] + \epsilon_o \cdot \text{orders}[idx] + \epsilon_{b}[idx]\]
\[\text{type} = \arg\max_t v'_t\]
where \\(\epsilon_o\\) is the soft margin of order and \\(\epsilon_b\\) is the base preference soft margin [10, 11].

## 3. PyTorch Pseudocode (Algorithm 1)

```python
import torch

def Degradation_Discriminator(X, cls, ep_o, orders, ep_b):
    # X: Input image
    # cls: Classifier network for degradation detection
    # ep_o: Soft margin of order (float)
    # orders: List of degradation orders for each basis
    # ep_b: Soft margin list per basis
    v = cls(X)  # Probabilities/logits vector
    for idx in range(len(orders)):
        v[idx] += ep_o * orders[idx] + ep_b[idx]
    return v.argmax(dim=-1)

def Chain_of_Restoration(X, M, cls, clean_idx, ep_o, orders, ep_b):
    # M: Pre-trained Multi-task Restoration Model
    # clean_idx: Class index for clean state (e.g., n+1)
    type_idx = Degradation_Discriminator(X, cls, ep_o, orders, ep_b)
    
    while type_idx != clean_idx:  # Loop until DD identifies clean image
        X = M(X, type_idx)        # Apply single-step restoration
        type_idx = Degradation_Discriminator(X, cls, ep_o, orders, ep_b)
        
    return X  # Fully restored image

```

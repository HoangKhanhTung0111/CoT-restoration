# SKILLS &amp; PROJECT INSTRUCTIONS: Efficient Universal Image Restoration (CoT-NAFNet)

## 1. Project Context &amp; High-Level Goal
This project aims to solve the **Composite Degradation / Degradation Coupling** problem in Universal Image Restoration (UIR) (where an image suffers from multiple overlapping corruptions, e.g., low-light + rain + haze).

We aim to combine two key ideas:
1. **NAFNet (Baseline Model):** An ultra-lightweight, high-efficiency CNN-based restoration network (`./nafnet` folder).
2. **Chain-of-Thought / Chain-of-Restoration Concept (from CoTIR &amp; UIR):** Internalizing or sequencing the restoration process into "Thinking -&gt; Planning -&gt; Action" steps to disentangle coupled degradations without heavy computational burden (`./cotir` folder).

**Primary Objective:** Develop a reproducible **CoT-NAFNet** research pipeline on Kaggle. First validate the inputs, checkpoints, tensor shapes, memory, and experiment recording; then compare against a NAFNet baseline using the same split, seed, pretrained initialization, loss, and training budget. Runtime is measured rather than forced into a two-hour MVP window. Metric improvement is an experimental hypothesis, not a guarantee.

---

## 2. Directory Structure

```

/project_root
├── skills.md <-- You are reading this file
├── docs/ <-- Converted markdown papers (CoTIR, UIR, AnyIR)
├── NAFNet/ <-- Modified NAFNet repository (Kaggle-ready)
├── CoTIR/ <-- Official CoTIR repository
├── notebook/ <-- NEW: Kaggle notebook references and outputs
│   └── nafnet.ipynb <-- Previous successful NAFNet run. Reference this for Kaggle environment setup and cell execution order.
├── notebooks/
│   └── kaggle_cot_nafnet.ipynb <-- Import this directly from GitHub; no per-cell copy/paste.
└── hybrid_cot_nafnet/ <-- NEW: Modules and integration scripts
    ├── modules/ <-- Custom CoT-Adapters / Gated Modules
    ├── datasets/ <-- Data Loaders for Composite Degradation
    ├── train_kaggle.py <-- Fast training script optimized for Kaggle
    └── evaluate.py <-- Inference & PSNR/SSIM metrics script

```

---

## 3. Technical Strategy (Approved)

Implement only **Strategy A: Lightweight CoT-inspired NAFNet (Single-Pass)**. Strategy B and all multi-pass restoration loops are out of scope.

* **Concept:** Attach one lightweight degradation-reasoning adapter to NAFNet's bottleneck and use it to modulate both bottleneck and skip features.
* **Mechanism:**
  1. *Thinking:* Extract content/degradation embeddings with depthwise convolution and supervise a four-label vector (`low`, `haze`, `rain`, `snow`). Do not claim feature disentanglement without an explicit training constraint.
  2. *Planning:* Produce zero-initialized affine channel gates for the bottleneck and every encoder skip connection.
  3. *Action:* Restore the image with the standard NAFNet decoder in one forward pass.
* **Efficiency constraint:** Keep the adapter below 0.5M parameters, use microbatching and tiled evaluation, and validate tensor shapes and dataset pairs before a long run.

---

## 4. Coding Agent Action Roadmap

When assigned tasks, follow these step-by-step instructions:

### Task 1: Repository Audit &amp; Code Inspection
* Inspect `nafnet/` to confirm its model definition (`NAFNet` architecture in PyTorch) and data pipeline.
* Inspect `cotir/` to understand how loss functions (Lagrangian / CoT loss) or dataset formatting are structured.

### Task 2: Module Construction (`hybrid_cot_nafnet/modules/`)
* Create a lightweight PyTorch module `CoTAdapter` or `GatedCoTBlock`:
  * Keep parameters minimal (&lt; 0.5M params).
  * Use depthwise separable convolutions and SimpleGate (consistent with NAFNet style).
* Inject this module into NAFNet's bottleneck without breaking existing pre-trained weight compatibility where possible.

### Task 3: Kaggle-Optimized Training Script (`train_kaggle.py`)
* Write a concise pure PyTorch training loop to minimize Kaggle dependency risk.
* Support Mixed Precision (`torch.cuda.amp.autocast()`) for maximum training speed.
* Use CosineAnnealingLR and AdamW optimizer.
* Add progress logging suitable for Kaggle notebooks.
* Save an experiment record for every run: config, environment, Git commit, dataset manifest, pretrained compatibility report, epoch log, summary, and checkpoints.

### Task 4: Evaluation &amp; Benchmarking (`evaluate.py`)
* Compute PSNR and SSIM on composite degradation test sets.
* Include latency (ms) and parameter count (`#Params`, `FLOPs`) calculations to prove efficiency.

---

## 5. Coding Guidelines &amp; Rules
* **PyTorch Best Practices:** Ensure all tensor operations preserve GPU memory.
* **No Unnecessary Dependencies:** Rely on standard packages available on Kaggle (`torch`, `torchvision`, `einops`, `timm`, `opencv-python`).
* **Modular Code:** Keep custom modules strictly isolated inside `hybrid_cot_nafnet/` to avoid corrupting working baseline code in `nafnet/`.
* **Data Safety:** Split CDD-11 by clean scene ID, verify every degraded/clean pair, and reject overlap between train, validation, and test IDs.
* **Clear Documentation:** Add concise docstrings explaining mathematical operations inside new PyTorch modules.
* **Math & Formula Clarification:** If any formula or mathematical notation in `docs/` appears corrupted/unclear due to PDF-to-Markdown conversion artifacts (e.g., unexpected `##` or broken LaTeX symbols), DO NOT guess the implementation. Ask the user for clarification or refer directly to the source code definitions in `cotir/` or `nafnet/`.



---

## 6. Kaggle Environment Config

Kaggle commands must use the exact paths below. Do not write placeholders such
as `/kaggle/input/<dataset-name>` and do not use `/kaggle/input` as a broad data
root when the exact CDD-11 path is known.

📁 input/
    📁 datasets/
        📁 mintesnotfikir/
            📁 cdd-11-30/
                📁 CDD-11_test/
                    📁 low_haze_snow/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 haze_rain/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 snow/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 low/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 rain/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 low_haze_rain/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 clear/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 haze_snow/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 haze/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 low_rain/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 low_snow/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                    📁 low_haze/
                        📄 00008.png
                        📄 00018.png
                        📄 00013.png
                        ... và 2 file khác
                📁 CDD-11_train/
                    📁 low_haze_snow/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 haze_rain/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 snow/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 low/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 rain/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 low_haze_rain/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 clear/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 haze_snow/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 haze/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 low_rain/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 low_snow/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
                    📁 low_haze/
                        📄 00122.png
                        📄 00038.png
                        📄 00009.png
                        ... và 22 file khác
        📁 hoangkhanhtung/
            📁 sidddata/
                📁 SIDD/
                    📁 val/
                        📁 input_crops.lmdb/
                            📄 lock.mdb
                            📄 data.mdb
                            📄 meta_info.txt
                        📁 gt_crops.lmdb/
                            📄 lock.mdb
                            📄 data.mdb
                            📄 meta_info.txt
            📁 goprodata/
                📁 GoPro/
                    📁 test/
                        📁 target.lmdb/
                            📄 lock.mdb
                            📄 data.mdb
                            📄 meta_info.txt
                        📁 input.lmdb/
                            📄 lock.mdb
                            📄 data.mdb
                            📄 meta_info.txt
            📁 nafnetmodel/
                📄 NAFNet-GoPro-width64.pth
                📄 NAFNet-GoPro-width32.pth
                📄 NAFNet-SIDD-width64.pth
                📄 NAFNet-SIDD-width32.pth


The CDD-11 input above is the small 30-pair-per-degradation subset. Label all
results accordingly and request the full CDD-11 dataset before making
publication-level claims.

The four `nafnetmodel` checkpoints live only on Kaggle. Inspect their real tensor
keys and shapes using `hybrid_cot_nafnet.audit_kaggle`; local/GitHub configs are
supporting evidence, not a substitute for auditing the actual files.

Before editing, testing, training, or changing Git state, give a realistic time
range. Once approved, move to the next approved task as soon as the previous task
finishes; never wait for the estimate to elapse.

`notebook/nafnet.ipynb` benchmarks four pretrained models on partial GoPro/SIDD
test data. It is not a NAFNet baseline result on CDD-11.

When designing CoTAdapter, reference docs/CoTIR\_adapter\_summary.md for parameter equations. Multi-pass CoR inference is intentionally excluded from the approved research direction.


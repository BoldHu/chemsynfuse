# ChemSynFuse

Core implementation of **ChemSynFuse: Domain-Adaptive Heterogeneous Information Fusion for Predictive Modeling of Chemical Synthesis**.

The model combines a bidirectional Mamba molecular encoder, typed table attention, hierarchical fusion, and conditional domain adaptation. It supports classification and regression.

## Installation

Validated with Python 3.10 and CUDA-enabled PyTorch 2.0.1. Molecular encoding requires the real `mamba-ssm` CUDA backend and a compatible CUDA build toolchain.

```bash
pip install -r requirements.txt
```

## Contents

- `molecular.py`, `pretrain.py`: molecular encoder and pretraining objective.
- `table.py`, `fusion.py`, `model.py`: table encoder, fusion and prediction head.
- `adaptation.py`, `domain_probe.py`: conditional alignment and adaptive mixing.
- `trainer.py`: warmup, AdamW training and EMA checkpoint selection.

## Usage

```python
from chemsynfuse import ChemSynFuse, ModelSchema, Trainer, TrainingConfig

# Dimensions must match your externally prepared feature tensors.
schema = ModelSchema(fields=20, categories=30, metadata_dim=16, summary_dim=8)
model = ChemSynFuse(schema, kind="classification", classes=4).cuda()
trainer = Trainer(model, TrainingConfig(learning_rate=3e-4))
# trainer.step(source_features, source_labels, target_features)
```

Feature dictionaries contain `values`, `states`, `available`, and `categories` shaped `[B,F]`; shared `types` and `sections` shaped `[F]`; `molecules` shaped `[B,M,256]`; `molecule_meta` shaped `[B,M,metadata_dim]`; `molecule_mask` shaped `[B,M]`; and `molecule_summary` shaped `[B,summary_dim]`. Masks are boolean and IDs are integers. Even field types are numerical; odd types are categorical. Section 0 represents recipe inputs and section 1 conditions. Observation-state IDs are defined in `constants.py`; observed zeros remain available. Use a masked placeholder slot for records without molecules.

Molecular features come from a frozen `MolecularPretrainer.encode()` using externally supplied pretrained weights and token IDs. Padding/BOS/EOS IDs are 0/3/2. `pretraining_step` accepts two tokenized molecular views and reconstruction targets.

For regression, set `kind="regression"` and pass a `RegressionNeighborhoods` instance to `Trainer`, initialized with source-training-only `mean`, `scale`, `anchors`, `bandwidth`, `minimum`, and `maximum`. The supervised objective is twice the Huber loss on standardized residuals; classification uses ordinary cross-entropy. `Trainer.fit` selects EMA weights using source-validation macro-F1 or RMSE. Its probe callback calls `update_omega` on original source-training and unlabeled target-adaptation features with group identifiers. Target evaluation samples are not training inputs.

This repository contains the core model and tensor-level training API. Datasets, preprocessing, baselines, pretrained weights, experiment results and internal reports are not included.

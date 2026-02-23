# LEAD: Learning Decomposition for Source-free Universal Domain Adaptation

CVPR 2024 paper implementation for Source-free Universal Domain Adaptation (SF-UniDA).

## Training Commands

Two-stage process: source model training → target adaptation (source data unavailable at stage 2). No unit tests or linters configured.

```bash
# Stage 1: Train source model (single domain)
python train_source.py --dataset Office --s_idx 0 --target_label_type OPDA --epochs 50 --lr 0.01

# Stage 2: Target adaptation (auto-loads source checkpoint from checkpoints/)
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --lam_psd 0.30 --target_label_type OPDA

# Full scenario runs (all domain pairs):
bash scripts/train_source_OPDA.sh [gpu_id] [seed]
bash scripts/train_target_OPDA.sh [gpu_id] [seed]
```

Scripts: `scripts/train_{source|target}_{OPDA|OSDA|PDA}.sh` — each iterates all domain pairs per dataset. They accept optional positional args for GPU ID (default 0) and random seed (default 2021). `scripts/reproduce_office31.sh` is a master script running all 3 scenarios for Office-31. Validation is by H-score output during training.

On macOS, use `--num_workers 0` to avoid multiprocessing issues (the reproduce script sets `NW=0` for this reason). Use `python3` explicitly on macOS; upstream scripts use `python`.

### Key Hyperparameters by Dataset

| Dataset | `lam_psd` | `lr` (source) | `lr` (target) | `epochs` (src/tgt) | `embed_feat_dim` |
|-----------|-----------|---------------|---------------|---------------------|------------------|
| Office | 0.30 | 0.01 | 0.001 | 50 / 50 | 256 |
| OfficeHome | 2.00 | 0.01 | 0.001 | 50 / 50 | 256 |
| VisDA | 1.00 | 0.001 | 0.0001 | 10 / 30 | 256 |
| DomainNet | 2.00 | 0.01 | 0.0001 | 50 / 10 | 1024 |

## Architecture

The model (`model/SFUniDA.py`) has three components:
- **backbone_layer**: Pretrained ResNet50 (or VGG) feature extractor
- **feat_embed_layer**: Linear → BatchNorm projecting to `embed_feat_dim`
- **class_layer**: Weight-normalized linear classifier

Forward pass returns `(embed_feat, logits)`. During target adaptation, `class_layer` weights are frozen (`requires_grad=False`) and SVD-decomposed into known/unknown subspace bases.

Weight initialization: Conv2d uses Kaiming uniform, Linear uses Xavier normal, BatchNorm uses Normal(1.0, 0.02).

### LEAD Algorithm (train_target.py)

The core method `obtain_LEAD_pseudo_labels` performs:
1. SVD decomposition of source classifier weights → known/unknown subspace bases (computed once at epoch 0, stored in module-level globals)
2. C_t estimation via t-SNE + KMeans + silhouette score (every 10 epochs)
3. Feature projection onto known/unknown subspaces; unknown-space norm indicates "private-ness"
4. Gaussian Mixture Model on unknown-space norms → two-component boundary
5. Instance-level decision boundaries via source/target prototype similarity + per-class thresholds
6. Student's t-distribution weighting for pseudo-label confidence

Three losses: `L_ce` (pseudo-label CE, weighted by `lam_psd`), `L_reg` (subspace projection regularization), `L_con` (KNN consistency, k=4 hardcoded).

Online bank updates: `pred_cls_bank` and `embed_feat_bank` are updated in-place each iteration, enabling online KNN computation during training.

### Optimizer & Scheduler

- SGD with momentum=0.9, weight_decay=1e-3, nesterov=True
- Backbone LR = 0.1× base LR; embed/classifier layers use full LR
- Polynomial decay: `lr × (1 + 10 × progress)^(-0.75)`
- Source training uses `CrossEntropyLabelSmooth` (epsilon=0.1 for "smooth", 0.0 for "vanilla")

## Dataset Configuration

- Datasets in `./data/{Office,OfficeHome,VisDA,DomainNet}/`
- Each subdomain has `image_unida_list.txt`: `<relative_path> <integer_label>` per line
- Class ordering: `[shared | source_private | target_private]`; target-private remapped to `source_class_num` (universal "unknown" index)
- Class splits defined in `config/model_config.py` per dataset and scenario (OPDA, OSDA, PDA)
- Domain indices: Office={0:Amazon, 1:Dslr, 2:Webcam}; OfficeHome={0:Art, 1:Clipart, 2:Product, 3:Realworld}; DomainNet={0:Painting, 1:Real, 2:Sketch}
- Dataset returns 4-tuple `(img_train, img_test, label, index)` — dual augmentations of same image
- Transforms: Train uses Resize(256)→RandomCrop(224)→RandomHFlip→Normalize; Test uses Resize(256)→CenterCrop(224)→Normalize

## Conventions and Gotchas

### Do Not Remove
- `torch.multiprocessing.set_sharing_strategy('file_system')` at module load in both training scripts
- `torch.multiprocessing.set_start_method('fork', force=True)` when CUDA is unavailable (enables MPS)
- Module-level globals `known_space_basis` / `unknown_space_basis` in `train_target.py` (persist SVD bases across epochs)
- Module-level globals `best_score` / `best_coeff` in `train_target.py` (track C_t estimation across training)

### Checkpoint Paths
- Source: `checkpoints/{dataset}/source_{s_idx}/source_{train_type}_{label_type}/latest_source_checkpoint.pth`
- Target: `checkpoints/{dataset}/s_{s_idx}_t_{t_idx}/{label_type}/{train_type}_psd_{lam_psd}/` (appends `_{note}` if `--note` set)
- `train_target.py` copies itself and `net_utils.py` into the target save_dir for reproducibility

### Evaluation
- Primary metric: H-score = harmonic mean of known accuracy and unknown accuracy (`compute_h_score()` in `utils/net_utils.py`)
- `w_0` (default 0.55) is the entropy threshold for open-set rejection at test time (note: `compute_h_score` has its own default of 0.5, but `args.w_0` overrides it)
- For PDA/CLDA scenarios, best model selected by `known_acc`; all others by `h_score`

### Special Cases
- Office-31 OSDA: target-private classes offset by +10 after source-private in `dataset/dataset.py`
- `preload_flg` pre-loads and caches resized images only for datasets containing "Office" in the name
- DomainNet overrides `embed_feat_dim` to 1024 in `config/model_config.py`
- `--test` flag switches the logger to append mode (for evaluation on existing log files)
- `source_train_type` defaults to `"smooth"` (label smoothing ε=0.1); `"vanilla"` uses ε=0.0

## Other Directories

- `jittor_version/`: Full port of the codebase to the [Jittor](https://github.com/Jittor/jittor) framework (separate training scripts, model, dataset). Not used by the main PyTorch pipeline.

## Environment

```bash
conda env create -f environment.yml
conda activate con_110
```

Key deps: PyTorch, torchvision, scikit-learn (KMeans, GMM, t-SNE, silhouette), tqdm, wandb. `get_device()` in `utils/net_utils.py` auto-selects CUDA > MPS > CPU.

# IndustrialGuard AI

Industrial visual anomaly detection research project. Phase 1 audits MPDD and MVTec AD; Phase 2 provides the archive-aware PyTorch data pipeline; Phase 3 implements PaDiM, PatchCore, and EfficientAD interfaces, training/inference scripts, thresholds, and evaluation metrics. Training is always an explicit command. No full-scale experiment starts automatically.

MPDD is reported as the primary mechanical-component benchmark. MVTec AD is a broader complementary industrial anomaly benchmark. Their test data and aggregate scores are kept separate.

## Data and experimental protocol

- MPDD’s verified archive layout is `MPDD/<category>/{train/good,test/good,test/<defect>,ground_truth/<defect>}`.
- MVTec AD’s verified layout has category roots with official `train/good`, `test/good`, `test/<defect>`, and `ground_truth/<defect>` paths.
- Only normal `train/good` records fit PaDiM/PatchCore and EfficientAD. A fixed-seed 10% validation subset is derived per category from those normal training records. The official `test` split is unchanged and used only after fitting and validation threshold selection.
- The complete split procedure, model-specific methods, normalization, thresholds, metrics, and limitations are documented in [experimental_protocol.md](reports/model_development/experimental_protocol.md).

Set archive paths in the environment; checked-in configuration has no personal paths:

```powershell
$env:INDUSTRIALGUARD_MPDD_ARCHIVE = 'D:/path/to/MPDD.zip'
$env:INDUSTRIALGUARD_MVTEC_ARCHIVE = 'D:/path/to/mvtec_anomaly_detection.tar.xz'
```

Install the optional model and evaluation dependencies:

```powershell
python -m pip install -e ".[models,evaluation,dev]"
```

Torchvision supplies pretrained ResNet backbones for PaDiM and PatchCore. Anomalib supplies the maintained EfficientAD model implementation. EfficientAD training additionally requires an ImageNet-pretrained teacher checkpoint and an ImageNet penalty-image directory. These resources are not downloaded automatically.

## Models

- **PaDiM** extracts ResNet `layer1`–`layer3` features, estimates a regularized Gaussian per spatial location from normal training images, and scores Mahalanobis distance maps.
- **PatchCore** extracts normal patch features, constructs a bounded coreset memory bank, and scores nearest-neighbor patch distances.
- **EfficientAD** uses Anomalib’s PDN teacher/student and autoencoder, including its teacher-student, ImageNet penalty, reconstruction, and consistency losses. It needs teacher weights and penalty data; a random feature extractor is not used as a substitute.

Model configs are in `configs/models/`. They select MPDD `bracket_black` by default; pass `--dataset mvtec --category screw` to select the requested MVTec smoke category. The full category set can be passed by repeating `--category` or by editing an experiment config. Each category is fitted independently.

### Train or fit

```powershell
python scripts/train_model.py --config configs/models/padim.yaml --allow-pretrained-download
python scripts/train_model.py --config configs/models/patchcore.yaml --allow-pretrained-download
```

For MVTec screw:

```powershell
python scripts/train_model.py --config configs/models/padim.yaml --dataset mvtec --category screw --allow-pretrained-download
python scripts/train_model.py --config configs/models/patchcore.yaml --dataset mvtec --category screw --allow-pretrained-download
```

EfficientAD requires explicit local training resources:

```powershell
python scripts/train_model.py --config configs/models/efficientad.yaml --teacher-checkpoint 'D:/weights/pretrained_teacher_small.pth' --imagenet-penalty-dir 'D:/datasets/imagenet/train'
```

The training script saves a per-model, per-dataset, per-category PyTorch checkpoint and a metadata JSON under `checkpoints/`. PaDiM and PatchCore estimate normal references rather than optimize epochs. EfficientAD uses the configured gradient steps. Existing torchvision weights are used from cache; add `--allow-pretrained-download` only when you explicitly authorize a weight download. Existing checkpoints are preserved unless `--overwrite` is passed.

### Smoke checks

Small capped runs check fitting and maps; random ResNet weights are explicitly selected to avoid presenting an untrained-backbone smoke as a benchmark. They produce no benchmark claim:

```powershell
python scripts/train_model.py --config configs/models/padim.yaml --dataset mpdd --category bracket_black --limit-per-split 4 --max-train-images 4 --image-size 64 --random-backbone
python scripts/train_model.py --config configs/models/patchcore.yaml --dataset mvtec --category screw --limit-per-split 4 --image-size 64 --random-backbone
```

EfficientAD’s architecture can produce finite maps, but a dataset training smoke is skipped until a pretrained teacher checkpoint and separate ImageNet penalty data are supplied. The Anomalib architecture-only CPU check is not a trained model result.

### Evaluate

```powershell
python scripts/evaluate_model.py --config configs/models/padim.yaml --checkpoint checkpoints/padim/mpdd/bracket_black/model.pt --dataset mpdd --category bracket_black --overwrite
python scripts/evaluate_model.py --config configs/models/patchcore.yaml --checkpoint checkpoints/patchcore/mvtec/screw/model.pt --dataset mvtec --category screw --overwrite
```

Evaluation derives category-specific image and pixel thresholds from normal validation scores (99.5th percentile by default), then applies them to the unchanged official test split. Test metrics include image AUROC/AP; pixel AUROC/AP/AUPRO; thresholded Dice, IoU, precision, recall, and F1; and latency/FPS. Dataset summaries macro-average available per-category metrics separately by dataset. Missing positive/negative classes produce `null` for undefined ranking metrics. Results are written under `reports/model_development/results/`.

For programmatic inference, instantiate with `industrialguard.models.create_model(...)`, fit or load a checkpoint, then call `model.predict(batch["image"])`; the result contains `image_score` and `anomaly_map` tensors. Evaluation thresholds are saved in checkpoint extras and recalibrated from validation by the evaluation command, never from test data.

## Data cache

The data module reads verified inventories and caches only records selected for the current dataset/category/split under `.cache/datasets/`. ZIP entries are read individually. TAR.XZ selection streams through the archive when uncached items are requested, while writing only selected files. Cache entries retain category/split paths and have SHA-256 manifest verification. `.cache/` is gitignored; full-category experiments can consume substantial disk space. Remove `.cache/datasets/` to reclaim it. Source archives stay outside the project and are never changed.

## Tests and quality checks

```powershell
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/industrialguard
```

Ordinary unit tests use synthetic data and CPU inference. The real data cache is used only by explicit smoke/training commands.

## Dependencies and licenses

The project’s own model code uses PyTorch. Optional model dependencies include Torchvision and Anomalib; evaluation uses scikit-learn and SciPy. Anomalib is Apache-2.0; Torchvision is BSD-3-Clause. EfficientAD implementation provenance and license links are recorded in the [experimental protocol](reports/model_development/experimental_protocol.md). Pretrained weights and datasets retain their upstream terms; do not redistribute them without checking those terms.

## DVC

Keep the original archives outside Git. Phase 4 uses DVC only for audit reports, manifests, configs and reproducibility metadata. It does not track dataset archives or generated model/cache artifacts, and it does not configure a DVC remote.

## Phase 4: reproducible benchmark and MLOps

The benchmark runner creates one immutable attempt folder per dataset/category/model/configuration. It seeds Python, NumPy, PyTorch, CUDA and DataLoader workers; records the exact selected config; uses the existing deterministic training/validation split; calibrates thresholds on validation normals; and evaluates the unchanged official test records. A failed run is retained and a retry gets a new `attempt-NNN` directory. Completed identifiers are skipped. It never overwrites prior checkpoints or results.

MLflow uses a project-local SQLite database and artifact folder at `runs/mlflow/mlflow.db` and `runs/mlflow/artifacts`; no server or remote is configured. Runs are nested under one parent run per dataset invocation. Checkpoints, sanitized config, environment versions, training/evaluation logs, per-category metrics, threshold/confusion counts, and representative normal/anomalous images, masks and predicted maps are logged. Raw archive paths, local model-resource paths, credentials and user-profile paths are omitted from MLflow artifacts. Benchmark status and intermediate logs live under `runs/benchmark/`; final tables are written to `reports/benchmarks/aggregates/`. These generated files, caches and checkpoints are ignored by Git.

Install the project tooling in its local virtual environment:

```powershell
python -m venv --system-site-packages .venv
$env:PIP_CACHE_DIR = "$PWD/.cache/pip"
.\.venv\Scripts\python.exe -m pip install -e ".[models,evaluation,dev,mlops]"
```

Run a selected subset. Dataset and model must be explicit; choose a category or `--all-categories`, so invoking the runner cannot accidentally start all 63 model/category combinations:

```powershell
# One model/category
.\.venv\Scripts\python.exe scripts/run_benchmark.py --dataset mpdd --model padim --category bracket_black --device auto

# One model across a dataset
.\.venv\Scripts\python.exe scripts/run_benchmark.py --dataset mvtec --model patchcore --all-categories --device cuda --batch-size 8 --image-size 256

# All three models for one category
.\.venv\Scripts\python.exe scripts/run_benchmark.py --dataset mvtec --model all --category screw --device cuda

# The full 63-run selection is possible only when explicitly requested
.\.venv\Scripts\python.exe scripts/run_benchmark.py --dataset all --model all --all-categories --device cuda
```

PaDiM and PatchCore require torchvision ImageNet ResNet weights. The runner checks the local Torch cache and stops before fitting if those weights are absent; only add `--allow-pretrained-download` when you explicitly authorize that download. EfficientAD requires the teacher checkpoint and ImageNet penalty images as explicit paths. `--max-steps` can limit EfficientAD for a pilot; reduced-step runs are not full benchmark runs:

```powershell
.\.venv\Scripts\python.exe scripts/run_benchmark.py --dataset mpdd --model efficientad --category bracket_black --device cuda --teacher-checkpoint 'D:/industrialguard-resources/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_small.pth' --imagenet-penalty-dir 'D:/industrialguard-resources/efficientad/imagenette2/train'
```

The [Anomalib EfficientAD source](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/efficient_ad/lightning_model.py) uses a pretrained teacher bundle and Imagenette penalty images. It lists the teacher bundle URL and SHA-256, but does not state a byte size; [upstream license clarification remains unresolved](https://github.com/open-edge-platform/anomalib/issues/3710), so do not assume the repository's Apache-2.0 license applies to the weights. The [standard ImageNet terms](https://www.image-net.org/download) limit use to non-commercial research and education. Imagenette is an ImageNet subset and should be treated under those restrictions unless separate rights are confirmed. For non-commercial research, the upstream-controlled download procedure is:

```powershell
$resourceDir = 'D:/industrialguard-resources/efficientad'
New-Item -ItemType Directory -Force $resourceDir | Out-Null
Invoke-WebRequest 'https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip' -OutFile "$resourceDir/efficientad_pretrained_weights.zip"
Get-FileHash "$resourceDir/efficientad_pretrained_weights.zip" -Algorithm SHA256
# Expected upstream SHA-256: c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a
Expand-Archive "$resourceDir/efficientad_pretrained_weights.zip" "$resourceDir/weights" -Force

# Imagenette is a separate, user-triggered download; reserve about 3 GB for the archive and extracted files.
Invoke-WebRequest 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz' -OutFile "$resourceDir/imagenette2.tgz"
tar -xzf "$resourceDir/imagenette2.tgz" -C $resourceDir
```

The ImageNet site requires registration and agreement to its terms; its [ILSVRC2012 training archive is 138 GB](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) and is not required when using upstream's Imagenette penalty set. The 3 GB Imagenette space budget is an estimate, not a measurement. The teacher release does not publish a byte size; check the release asset before downloading and allow additional headroom. No command above runs automatically.

Before benchmark use, create the baseline environment fingerprint and initialize local DVC metadata:

```powershell
.\.venv\Scripts\python.exe scripts/write_reproducibility_metadata.py
.\.venv\Scripts\python.exe scripts/setup_dvc.py
```

DVC versions `configs/`, the dataset audit reports/manifests, the experimental protocol and `reports/reproducibility/environment.json` in the local `.dvc/cache`. It does not add source archives, `.cache/`, checkpoints, or benchmark outputs. No DVC remote exists. In restricted Windows environments, DVC's internal site cache can be placed inside the ignored project cache for commands:

```powershell
$env:DVC_SITE_CACHE_DIR = "$PWD/.cache/dvc/site"
.\.venv\Scripts\python.exe -m dvc status
```

The generated `.dvc` pointer files and `.dvc/config` are intended for Git; no commit is created automatically. A future remote should be agreed and configured explicitly, for example `dvc remote add -d <name> <approved-url>` after credentials and access policy are settled. Do not put secrets in `.dvc/config`.

For a local-only tracking UI, start it manually from the project root when desired:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri "sqlite:///$($PWD.Path.Replace('\','/'))/runs/mlflow/mlflow.db" --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`. The runner itself uses the local SQLite URI directly and does not start a server.

## Phase 5: Colab first benchmark

The prepared [Colab notebook](notebooks/colab/01_padim_mpdd_bracket_black.ipynb) runs only PaDiM on MPDD `bracket_black` with seed 42 and 256-pixel inputs. It uses the existing data module, benchmark runner, evaluator, and local MLflow configuration. The verified inventory contains 289 official normal training images; with the configured deterministic 10% validation split, expect 260 training and 29 validation images, plus the unchanged official test split of 32 normal and 47 anomalous images with 47 masks. The notebook checks these counts against the checked-in inventory and stops on a mismatch.

The repository has no GitHub remote configured. The notebook therefore supports a user-supplied source ZIP as well as cloning only when a repository URL is explicitly set. Build a source ZIP from this checkout without `.git`, `.venv`, `.cache`, datasets, checkpoints, or MLflow files; include the DVC-hydrated `configs/`, `reports/dataset_audit/`, `reports/model_development/experimental_protocol.md`, and `reports/reproducibility/environment.json` payloads because there is no DVC remote to restore them from. Then upload the ZIP in the source cell. In Colab, mount Drive and put the original MPDD archive at `MyDrive/IndustrialGuardAI/datasets/MPDD (1).zip`; the archive is read-only, while selected verified members are cached under `/content/industrialguard_cache/`.

The notebook checks the active Colab Python, PyTorch, Torchvision, CUDA runtime, GPU, disk, and ResNet18 ImageNet weight cache. It never silently downloads weights. If the `ResNet18_Weights.DEFAULT` checkpoint is absent, follow the notebook's explicit, opt-in download cell and verify its SHA-256 before proceeding. The experiment uses the documented normal-only validation 99.5th-percentile threshold; test labels and masks are used only for final evaluation. Benchmark execution is gated behind a cell setting that defaults to `False` and requires GPU, archive, dependencies, split counts, and pretrained weights to pass first.

The runner's SQLite MLflow database and artifact directory are local to the Colab runtime to avoid relying on SQLite locking over Drive. After the run, the notebook exports the completed attempt, MLflow run artifacts, and a retrieval summary to `MyDrive/IndustrialGuardAI/artifacts/experiments/<experiment-id>/`. It does not copy the dataset cache. This preserves experiment outputs after runtime disconnection; the SQLite tracking database itself is runtime-local, so the exported MLflow artifact bundle and run summary are the portable record. A downloaded experiment folder on Windows includes the checkpoint, metrics, configuration, logs, and representative actual prediction visualizations for local inspection and inference.

### Resource planning (estimates, not measurements)

The active Windows virtual environment has CPU-only PyTorch (`torch 2.14.0+cpu`) and no CUDA support. The host exposes an NVIDIA GeForce MX550 with 2 GiB VRAM through `nvidia-smi`, below the project's suggested GPU budget; no full benchmark resource consumption has been measured. Start at 256×256. Use batch 8 for PaDiM/PatchCore on a GPU with at least 8 GB VRAM, reducing to 4 or 1 if memory is tight. EfficientAD requires batch 1. At 256-pixel resolution, budget roughly 4–8 GB VRAM for PaDiM/PatchCore and 2–4 GB for EfficientAD; actual peaks depend on GPU, resolution, and PyTorch/CUDA versions. CPU inference is supported but a full run may take days, particularly EfficientAD's configured 70,000 steps per category.

For time planning only, allow roughly 5–30 minutes per PaDiM/PatchCore category and 1–8 GPU-hours per EfficientAD category. Across 21 categories and all three models, that suggests approximately 25–190 GPU-hours, excluding installation, data extraction, and reruns. These are broad estimates, not measured results. Reserve 5–10 GB for checkpoints and duplicated MLflow checkpoint artifacts, plus the extracted image cache (dataset archive sizes and filesystem compression vary). Sequential runs reduce peak GPU memory; failed attempts retain outputs and therefore consume additional disk space.

Recommended first benchmark: PaDiM on MPDD `bracket_black`, seed 42, 256 resolution, batch size 8, one category. Verify the pretrained backbone cache and GPU memory first. Then run PatchCore on the same category/configuration, and proceed to EfficientAD only after its resource and license prerequisites are resolved.

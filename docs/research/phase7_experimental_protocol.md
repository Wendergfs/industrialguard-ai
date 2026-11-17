# Phase 7: controlled optimization protocol

## Current status

The independent labeled validation protocol is approved. Its Drive/Colab folder and manifest have not yet been provided, so no Phase 7 training or benchmark run has started. The MPDD official test archive and test split remain unchanged.

The PaDiM result in [`mpdd_bracket_black_padim_tesla_t4.json`](../benchmarks/mpdd_bracket_black_padim_tesla_t4.json) is retained as an exploratory historical baseline. Its test metrics will not be used to choose Phase 7 thresholds or hyperparameters. The local checkpoint files are capped random-backbone smoke tests and are not benchmark references.

The historical AUPRO value used a 200-quantile threshold grid and 4-connected regions. The Phase 7 evaluator now uses exact distinct-score thresholds and 8-connected regions, matching the Anomalib 2.7 reference behavior. Keep the old report unchanged; recalculate its AUPRO from the exported per-sample prediction maps and masks before comparing it with new runs.

## Data roles and separation

- **Fit:** official MPDD `train/good` records only. PaDiM and PatchCore reject anomalous fit batches; EfficientAD also requires a separate ImageNet penalty loader and a verified pretrained teacher.
- **Independent calibration:** user-provided labeled normals and anomalies, including binary masks for anomalous images. This data must be separate from the official MPDD test archive and representative of the intended inspection setting.
- **Official test:** untouched, used only after configs and threshold rules are locked. Its labels and masks must not select a model, hyperparameter, calibration method, or threshold.

The validation manifest is UTF-8 CSV with these columns:

```csv
sample_id,dataset,category,label,image_path,mask_path
validation-normal-001,mpdd-validation,bracket_black,0,images/normal/001.png,masks/normal/001.png
validation-defect-001,mpdd-validation,bracket_black,1,images/defect/001.png,masks/defect/001.png
```

Paths are relative to the supplied validation root. `sample_id` is stable and unique; labels are `0` (normal) or `1` (anomalous). Each anomaly requires a binary mask. Normal masks may be blank or omitted; a provided normal mask must be empty. Absolute paths, path traversal, missing files, duplicate IDs, and image/mask size mismatches are rejected. The loader reads but does not modify this data. It records a content fingerprint without putting source paths in the benchmark report.

## Threshold protocol

Candidate strategies are fixed before final test evaluation:

- Normal-only quantiles: 99%, 99.5%, and 99.9%, for image scores and pixel maps separately.
- Normal-only robust threshold: median plus a preregistered multiple of scaled MAD.
- Independent labeled validation: F1-selected thresholds and threshold/PR operating-point curves. A per-category threshold requires both normal and anomalous calibration examples for that category.

Every candidate's TP, FP, FN, TN, precision, recall, F1/Dice, and IoU is reported on calibration data. Test metrics are computed only after threshold selection and are reported for the locked choice; the test result is never fed back into tuning. If there is no agreed industrial recall floor or false-alarm budget, report the Pareto choices and do not declare one operating point universally best.

## Controlled run order

1. Reproduce and archive the historical PaDiM baseline configuration and its verified ResNet18 weight identity without changing the official test split.
2. Establish one bracket_black baseline per model. Run EfficientAD only after its teacher checkpoint and penalty data are available and fingerprinted.
3. Tune PaDiM in separate one-factor stages: map smoothing and image-score aggregation; feature dimension/reduction; covariance regularization; then resolution/backbone if the T4 memory preflight permits.
4. Tune PatchCore separately: patch stride, coreset ratio, memory size, then map smoothing/score aggregation.
5. Compare all locked candidates once on the official test. Expand to the remaining MPDD categories only after the bracket_black pipeline and artifacts pass review; MVTec is a separate dataset report.

Each invocation is limited to three model/category combinations unless `--allow-massive` is explicitly provided. A custom model config can be supplied for one model and one category with `--config`; generate tuning configs outside the checked-in DVC-managed `configs/` tree.

## Tracking and diagnostics

Runs are tagged `smoke_test`, `baseline`, `tuning`, or `final_benchmark`, and include the code revision (or source ZIP SHA-256), configuration and resource fingerprints, seed, threshold method/values, training-fit and inference durations, training/inference peak CUDA allocation, and image/pixel metrics. Calibration curves are calculated from calibration data. Test reports retain per-image IDs, image TP/FP/FN/TN status, and pixel confusion counts so false-positive and false-negative visualizations can be selected by outcome.

Do not store dataset samples, masks, credentials, or local source paths in Git. Keep checkpoints, prediction arrays, MLflow databases, and rendered diagnostics in the private experiment artifact store.

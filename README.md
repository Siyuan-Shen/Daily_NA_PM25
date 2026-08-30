v1.1.0 - 2026/02/17 Siyuan Shen {s.siyuan@wustl.edu}

## Changes from v1.0.0

### 1. Simulation Input: GCC → GCHP
- Replaced all `GC_*` channels (GCC-based) with `scsg_GCHP_*` channels (GCHP-based), credit to Yuanjian Zhang {yuanjian.z@wustl.edu}.
  - `GC_PM25` → `scsg_GCHP_PM25`, `GC_SO4` → `scsg_GCHP_SO4`, `GC_NH4` → `scsg_GCHP_NH4`, `GC_NIT` → `scsg_GCHP_NIT`, `GC_OM` → `scsg_GCHP_POA`, `GC_SOA` → `scsg_GCHP_SOA`, `GC_DST` → `scsg_GCHP_DST`, `GC_SSLT` → `scsg_GCHP_SSLT`.
- Added `GCHP_version` configuration variable (`CombinedAOD_GL_v20251119`) and corresponding GCHP input file paths in `Estimation_pkg/utils.py`.

### 2. Ratio-Calibrated AOD and Geophysical PM2.5
- Channel names `tSATAOD_Ratio_Calibration` and `tSATPM25_Ratio_Calibration` are renamed to `tSATAOD` and `tSATPM25` throughout the model config, net architecture, and wandb sweep config.
- Estimation input paths for `tSATAOD` and `tSATPM25` updated to use the `Ratio_Calibrated/` subdirectory structure.

### 3. Training Acceleration
- **Model compilation**: `torch.compile()` (PyTorch 2.0+) applied to daily models with a persistent compile cache (`/s.siyuan/my-projects2/torch_compile_cache`).
- **Automatic Mixed Precision (AMP)**: `autocast('cuda')` and `GradScaler` added to the training loop in `Training_pkg/TrainingModule.py`.
- **DataLoader optimization**: `pin_memory=True` enabled on train and validation DataLoaders.
- Environment variables (`MASTER_ADDR`, `MASTER_PORT`, `TORCH_COMPILE_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) moved to the top of `main.py` before all imports.

### 4. Training/Validation Separation in Cross-Validation
- Added `Use_saved_models_to_reproduce_validation_results` switch in all CV modules (Spatial, BLISCO, Random, TBO, Temporal).
- When enabled, training is skipped and saved models are loaded directly for validation, allowing reproduction of results without retraining.

### 5. TensorData Optimization
- `Training_pkg/TensorData_func.py`: zero-copy path when input data is already a `torch.Tensor`; dtype-aware (`float32`) conversion for numpy arrays to avoid unnecessary copies.

### 6. 3D CNN Output Channel Size Adjustment
- Default output channel sizes changed from `[128, 256, 512, 1024]` to `[64, 64, 128, 256]` in `Net_Architecture_config.py`.

### 7. Statistics NaN Handling Fix
- `Training_pkg/Statistic_func.py`: updated statistics functions to correctly handle NaN values in numpy arrays.

### 8. Model Loading Fix
- `Training_pkg/iostream.py`: all `torch.load` calls now use `map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')` to support loading GPU-trained models on CPU-only machines.

### 9. Wandb Sweep Config Narrowed
- Reduced hyperparameter search space: `wandb_sweep_count` reduced from 100 to 20; learning rate, batch size, and other search values narrowed based on prior tuning results.

---

## Post-release updates to v1.1.0

### 10. `NonNegMSE` Loss Function
- New loss type in `Training_pkg/Loss_func.py`: `loss = MSE + λ · mean(relu(-(pred × true_std + true_mean)))` — penalizes physically impossible negative PM2.5 predictions in denormalized (µg/m³) space.
- Default `Regression_loss_type` switched from `MSE` → `NonNegMSE` with `NonNegMSE_lambda = 5.0` in `config.py`.
- `true_y_mean` and `true_y_std` now propagated through `CNN3D_train` and all CV modules (`Random`, `Spatial`, `BLISCO`, `TBO`, `Temporal`, `Hyperparameter_Search`) so the loss can denormalize.
- `NonNegMSE_lambda` added to the wandb sweep parameter space in `wandb_config.py`.
- Training validation set size changed from hardcoded 100 to `config_batchsize × world_size` to ensure at least one full batch per GPU after `DistributedSampler` split.

### 11. `softplus_output` Post-processing
- New `softplus_output` flag added to `config.py` learning-objective settings (default `False`).
- When enabled, applies `log(1 + exp(x))` (softplus / `np.logaddexp(0, x)`) to the final output to enforce non-negativity.
- Applied in `Estimation_pkg/predict_func.py::map_data_final_output` and `Evaluation_pkg/data_func.py::Get_final_output`.

### 12. Multi-GPU NCCL Slowdown Fix
- Fixed a ~38x Day-2 GPU slowdown caused by NCCL's background auto-tuning competing with inference.
- `NCCL_ALGO=Ring` and `NCCL_PROTO=Simple` are now set via `os.environ.setdefault` before DDP setup to pin the collective algorithm and prevent the auto-tuning phase.
- A persistent `out_persistent` tensor is allocated once per run (instead of per date), so NCCL reuses the same registered memory address across all-reduces.
- 20-iteration NCCL warmup all-reduces (with realistic PM2.5-like data) are performed before the date loop to exhaust the tuning budget upfront.
- `dist.barrier()` + `torch.cuda.synchronize()` added after each all_reduce to fully quiesce NCCL proxy threads before the next date.
- `broadcast_buffers=False` set on DDP wrappers to reduce unnecessary synchronization.

### 13. `torch.compile` Warmup
- Warmup forward passes (batch sizes 65536, 1, then 1000/2000/5000) are executed before the date loop to JIT-compile CUDA kernels synchronously, preventing background recompilation from causing a 3–5× slowdown on Day 2.
- `torch._dynamo.reset()` called in `main.py` between wandb sweep runs to clear CUDA graph TLS state and avoid `AssertionError` on subsequent compilations.
- Added `hasattr(torch._inductor, 'config')` guard for `fx_graph_cache` to handle environments where the attribute may not exist.

### 14. Parallel Channel Loading
- `load_map_data` in `Estimation_pkg/iostream.py` now loads all input channels concurrently using `ThreadPoolExecutor` (up to 16 workers), reducing sequential I/O bottleneck.
- Output array dtype changed from `float64` → `float32` to halve memory usage.
- Per-channel print gated to rank 0 only.

### 15. Date-Sliding Data Cache for 3D CNN Inference
- Consecutive inference dates now reuse the previous day's data array: the cache is shifted in-place (slots 0..depth-2 ← 1..depth-1) and only the newest day is loaded, reducing redundant I/O by `(depth-1)/depth`.
- Non-consecutive dates (or the first date) fall back to loading all days in parallel via `ThreadPoolExecutor`.

### 16. Model Save Fix: Unwrap DDP / `torch.compile` Before Saving
- New `_get_saveable_model()` helper in `Training_pkg/iostream.py` strips DDP (`.module`) and `torch.compile` (`._orig_mod`) wrappers before `torch.save`.
- Prevents `AssertionError` inside `cudagraph_trees` when a compiled model is reloaded in a different execution context (e.g., a subsequent wandb sweep run).

### 17. Inference Overhaul
- `torch.no_grad()` replaced with `torch.inference_mode()` in both `cnn_mapdata_predict_func` and `cnn3D_mapdata_predict_func` for lower overhead.
- DataLoader replaced with direct tensor batching: `INFER_BATCH_SIZE = 65536` for 3D CNN, `4096` for 2D CNN, eliminating DataLoader worker overhead.
- Removed duplicate `model(image)` call (pre-AMP path) in `cnn_predict_3D`.

### 18. Per-Date Timing Diagnostics
- `[TIMING]` log line printed per inference date: `load / infer / reduce / save / total` wall-clock times.
- Per-row diagnostics: cumulative CPU prep time, GPU forward time, pixel count, NaN fill count.
- First-row GPU latency (`[DIAG]`) logged with `cuda.synchronize()` to detect when Day-2 slowness begins.

---

## Post-release updates to v2.0.0

### 19. `SCSG_GCHP_version` Configuration
- Added `SCSG_GCHP_version = 'GCHP.v13.2.1/stretchNA.output.noDOW.base'` to `config.py` for Aaron's stretched-grid GCHP data.
- Added `SCSG_GCHP_indir` path in `Estimation_pkg/utils.py`.
- Added 9 new entries in `inputfiles_table`: `scsg_GCHP_PM25`, `scsg_GCHP_SO4`, `scsg_GCHP_NH4`, `scsg_GCHP_NIT`, `scsg_GCHP_BC`, `scsg_GCHP_POA`, `scsg_GCHP_SOA`, `scsg_GCHP_DST`, `scsg_GCHP_SSLT`.

### 20. Log-Transform for Emission Channels
- **Motivation**: CEDS anthropogenic and dust/sea-salt offline emission channels are heavy-tailed (mostly ~0 with rare, large point sources), which let raw CEDS point-source cells leak through the MoCE base expert as punctate hotspots not seen in `tSATPM25`/GCHP fields at the same pixels — see hotspot investigation notes. Fix: `log10(x + eps)` applied to these channels before the existing per-channel Gaussian z-score normalization.
- `log_transformed_channels` (list) and `log_transform_eps` (dict) added to `Net_Architecture_config.py`, next to the `'MoCE-architecture'` block.
- **eps is a fixed constant per channel, not recomputed per training run.** Chosen instead of a single shared eps because the 8 channels span ~20 orders of magnitude in real signal, and instead of a dynamically-derived-every-run eps to keep it identical across training/testing/estimation regardless of which subset of dates a given run touches.
- **How eps is derived** (method, not to be confused with training-site statistics):
  1. For each channel, sample several dates spread across the full training period (used: 2018-03-15, 2020-11-15, 2022-07-15, 2024-07-05).
  2. For each sample date, load the **full continental map array** via `inputfiles_table(YYYY,MM,DD)` from `Estimation_pkg.utils` (NOT the per-site 3×5×5 training tensors in `Training_Datasets/.../3DCNN_centerwindow/`).
  3. Pool the nonzero values across all sample dates, take `median(nonzero)`.
  4. `eps = 1e-3 × median(nonzero)`.
  - **Full map, not training-site windows, deliberately**: training windows are 5×5 crops centered on ground monitoring stations, which skew heavily toward populated/monitored areas. Verified directly: training-window medians run ~60x higher than full-map medians for CEDS channels, and training windows had 0% exact-zero cells vs 6-10% on the full map. An eps sized off training-window data would be too large relative to the full estimation domain — quantified impact: for NH3 specifically, an eps derived from training-window median would floor 74% of the map's real nonzero signal into indistinguishable-from-zero.
  - **Per CEDS-version, not per species**: `v2023-04` and `v2025-04` CEDS releases differ in native spatial resolution before regridding to the 0.01°×0.01° NA grid, which measurably shifts the per-pixel distribution — every anthro species' median nonzero value is consistently ~2-3x higher in `v2023-04` than `v2025-04` on the same output grid. `log_transform_eps` is keyed by the exact versioned channel name (e.g. `NO_anthro_emi_v2025-04`), not just species, for this reason.
- **Current values** (`Net_Architecture_config.py`, computed 2026-08-11): CEDS v2025-04 — NO 6.03e-16, SO2 1.16e-16, OC 3.17e-18, BC 3.87e-18, NH3 1.13e-18, NMVOC 5.01e-17. CEDS v2023-04 — NO 1.12e-15, SO2 4.22e-16, OC 7.59e-18, BC 9.68e-18, NH3 3.50e-18, NMVOC 1.10e-16. Dust/sea-salt offline (Daily, currently active) — DST 3.39e-14, SSLT 6.34e-14; (Monthly, not currently used by any active MoCE channel list) — DST 2.98e-14, SSLT 1.04e-13.
- **If `CEDS_version`/`DUST_version`/`SSLT_version` changes to a version not already in `log_transform_eps`, recompute eps for it using the method above before training** — do not reuse an existing version's value, and do not let the lookup silently fall back to a default.
- **Wired into both training and estimation** (`apply_log_transform(data, channel_name)` in `Training_pkg/utils.py`, a no-op unless `channel_name` is in `log_transformed_channels`, and raises `KeyError` rather than silently falling back if a listed channel has no `log_transform_eps` entry):
  - `Training_pkg/data_func.py::CNN3DInputDatasets._load_aggregate_3DTrainingDatasets` applies it to each site's raw per-channel tensor immediately after loading, before concatenation and before `_derive_trainingdatasets_normalization_matrix` computes mean/std — so the training mean/std are derived from log-transformed values for these channels.
  - `Estimation_pkg/iostream.py::load_map_data` applies it to each channel's raw map array immediately after `np.load`, before the `train_mean`/`train_std` subtraction in `predict_func.py`.
  - `log_transformed_channels` is currently `[]` (transform inert everywhere) pending a training run to validate end-to-end; the channel list is commented out in `Net_Architecture_config.py` ready to re-enable.
- **Impact check** (raw z-score vs log+z-score, full-map pooled from the 4 sample dates above, mean/std from real training-window data): under the current raw (no-log) pipeline, CEDS point-source pixels on the map produce z-scores far outside the range the model was trained on; under log+z-score the same pixels land at bounded, well-separated-but-not-absurd z-scores, and the zero-cell z-score (`z(zero)`, from `eps`) stays clearly separated from real signal (-5.6 to -10.0 across channels) without collapsing the map's real dynamic range (p1-to-p99 spread preserved, ~9-12 z-units per channel). See `2026-08-11` analysis in project notes for the full raw-vs-log percentile/outlier table.

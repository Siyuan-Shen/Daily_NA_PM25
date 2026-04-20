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

### 19. `SCSG_GCHP_version` Configuration
- Added `SCSG_GCHP_version = 'GCHP.v13.2.1/stretchNA.output.noDOW.base'` to `config.py` for Aaron's stretched-grid GCHP data.
- Added `SCSG_GCHP_indir` path in `Estimation_pkg/utils.py`.
- Added 9 new entries in `inputfiles_table`: `scsg_GCHP_PM25`, `scsg_GCHP_SO4`, `scsg_GCHP_NH4`, `scsg_GCHP_NIT`, `scsg_GCHP_BC`, `scsg_GCHP_POA`, `scsg_GCHP_SOA`, `scsg_GCHP_DST`, `scsg_GCHP_SSLT`.

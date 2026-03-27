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

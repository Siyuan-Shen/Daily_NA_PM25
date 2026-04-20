import numpy as np
import gc
import os
import torch
from concurrent.futures import ThreadPoolExecutor as _TPE
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import time
from torch.amp import autocast
from Estimation_pkg.utils import *
from Estimation_pkg.data_func import get_extent_index, get_landtype
from Estimation_pkg.iostream import *
from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Split_Datasets_based_site_index,randomly_select_training_testing_indices,Get_final_output
from Evaluation_pkg.iostream import *
from Evaluation_pkg.Statistics_Calculation_func import calculate_statistics
from Model_Structure_pkg.CNN_Module import initial_cnn_network
from Model_Structure_pkg.ResCNN3D_Module import initial_3dcnn_net
from Model_Structure_pkg.utils import *

from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, learning_rate0 as config_learning_rate0
from Training_pkg.TensorData_func import Dataset_Val, Dataset
from Training_pkg.TrainingModule import CNN_train, cnn_predict, CNN3D_train, cnn_predict_3D
from Training_pkg.data_func import CNNInputDatasets, CNN3DInputDatasets
from Training_pkg.iostream import load_daily_datesbased_model
from multiprocessing import Manager
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cnn_mapdata_predict_func(rank, world_size,model,predict_begindate,predict_endate,total_channel_names,
                                evaluation_type, typeName,Area,extent,
                                train_mean, train_std, true_mean, true_std, width,height):
    nchannel = len(total_channel_names)
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting 3D CNN predict")
        # Your original CNN3D_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        print(f"[Rank {rank}] Exception occurred:\n{traceback.format_exc()}")
        raise e
    print(f"Rank {rank} process started.")

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    elif world_size > 1:
        ddp_setup(rank, world_size)
        device = rank
        model.to(device)
        model = DDP(model, device_ids=[device], broadcast_buffers=False)

    AllDates, int_AllDates = create_date_range(predict_begindate, predict_endate)
    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype('2020',extent)
    lat_infile = LATLON_indir + 'NA_SATLAT_0p01.npy'
    lon_infile = LATLON_indir + 'NA_SATLON_0p01.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    AOD_index = total_channel_names.index('tSATAOD')
    INFER_BATCH_SIZE = 4096

    with torch.inference_mode():
        model.eval()

        for date in int_AllDates:
            # Your original CNN3D_train logic goes here...
            YYYY,MM,DD = getGrg_YYYY_MM_DD(date)
            print(f"Processing date: {YYYY}-{MM}-{DD}")
            output = np.full((len(lat_index),len(lon_index)),-9999.0,dtype=np.float32)

            ## Load Map Data for the date
            YYYY, MM, DD = getGrg_YYYY_MM_DD(date)
            temp_map_data = load_map_data(total_channel_names, YYYY, MM, DD, rank)
            
            ## Convert to 3D CNN reading and predict
            for ix in range(len(lat_index)//world_size):
                ix = ix*world_size + rank
                land_index = np.where((landtype[ix,:] != 0) & (~np.isnan(temp_map_data[AOD_index, lat_index[ix], lon_index])))

                if ix % max(1, len(lat_index) // 10) == 0:
                    print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
                if len(land_index[0]) > 0:
                    temp_input = np.zeros((len(land_index[0]), nchannel, width, width), dtype=np.float32)
                    for iy in range(len(land_index[0])):
                        temp_input[iy,:,:,:] = temp_map_data[:,int(lat_index[ix] - (width - 1) / 2):int(lat_index[ix] + (width + 1) / 2), int(lon_index[land_index[0][iy]] - (width - 1) / 2):int(lon_index[land_index[0][iy]] + (width + 1) / 2)]

                    temp_input -= train_mean
                    temp_input /= train_std

                    t_input = torch.from_numpy(temp_input)
                    outputs = []
                    for start in range(0, len(t_input), INFER_BATCH_SIZE):
                        batch = t_input[start:start + INFER_BATCH_SIZE].to(device, non_blocking=True)
                        outputs.append(model(batch).squeeze(1).cpu().numpy())
                    output[ix, land_index[0]] = np.concatenate(outputs)

            if world_size > 1:
                out_t = torch.from_numpy(output).to(device)
                dist.all_reduce(out_t, op=dist.ReduceOp.MAX)
                output = out_t.cpu().numpy()
            ## Save the output
            if rank == 0:
                output = map_data_final_output(output,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD, softplus_output)
                save_Estimation_Map(mapdata = output, outdir = data_recording_outdir,
            file_target='Map_Estimation', typeName=typeName, Area=Area, YYYY=YYYY, MM=MM, DD=DD, nchannel=nchannel,
            width=width, height=height
            )
    if world_size > 1:
        dist.barrier()
        destroy_process_group()

    return

def cnn3D_mapdata_predict_func(rank, world_size,model,predict_begindate,predict_endate,total_channel_names,
                                evaluation_type, typeName,Area,extent,
                                train_mean, train_std, true_mean, true_std, width,height,depth):
    nchannel = len(total_channel_names)
    AOD_index = total_channel_names.index('tSATAOD')
    print('world_size: {}'.format(world_size))
    try:
        print(f"[Rank {rank}] Starting 3D CNN predict")
        # Your original CNN3D_train logic goes here...
    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        import traceback
        print(f"[Rank {rank}] Exception occurred:\n{traceback.format_exc()}")
        raise e
    print(f"Rank {rank} process started.")

    cache_dir = '/s.siyuan/my-projects2/torch_compile_cache'
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
    os.environ['TORCH_COMPILE_CACHE_DIR'] = cache_dir
    torch.set_float32_matmul_precision('high')
    if hasattr(torch._inductor, 'config'):
        torch._inductor.config.fx_graph_cache = True

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # dynamic=True handles variable batch sizes (last batch per row differs from 2000)
        # without triggering recompilation for each new shape
        model = torch.compile(model, dynamic=True)
    elif world_size > 1:
        # Force NCCL to use Ring algorithm with Simple protocol.
        # Without this, NCCL auto-selects its algorithm on the first real all_reduce
        # (day 1's result) and runs GPU-side benchmark kernels in background proxy
        # threads.  Those benchmarks compete with day 2 inference, causing a ~38x
        # GPU slowdown.  Pinning to Ring/Simple disables the auto-tuning phase.
        os.environ.setdefault('NCCL_ALGO', 'Ring')
        os.environ.setdefault('NCCL_PROTO', 'Simple')
        ddp_setup(rank, world_size)
        device = rank
        model.to(device)
        #model = torch.compile(model, dynamic=True)
        model = DDP(model, device_ids=[device], broadcast_buffers=False)

    AllDates, int_AllDates = create_date_range(predict_begindate, predict_endate)
    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype('2020',extent)
    lat_infile = LATLON_indir + 'NA_SATLAT_0p01.npy'
    lon_infile = LATLON_indir + 'NA_SATLON_0p01.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    half_width = int((width - 1) / 2)
    half_height = int((height - 1) / 2)

    # Data cache: reuse previous days' data across consecutive dates
    # cached_map_data[:, slot, :, :] where slot 0=oldest … depth-1=newest
    cached_map_data = None
    cached_last_date = None   # the date stored in slot depth-1 of cached_map_data

    with torch.inference_mode():
        model.eval()

        INFER_BATCH_SIZE = 65536
        # Warmup: force torch.compile to JIT-compile CUDA kernels for both the
        # full batch size and a tail batch of 1, so no recompilation occurs mid-run.
        _dummy_full = torch.zeros(INFER_BATCH_SIZE, nchannel, depth, width, height,
                                  device=device, dtype=torch.float32)
        _dummy_tail = torch.zeros(1, nchannel, depth, width, height,
                                  device=device, dtype=torch.float32)
        with autocast('cuda'):
            model(_dummy_full)
            model(_dummy_tail)
        del _dummy_full, _dummy_tail
        # Additional warmup: trigger symbolic kernel compilation for actual batch sizes.
        # The two calls above compiled specialised kernels for B=65536 and B=1; a call
        # with an intermediate size forces Dynamo to produce the general dynamic kernel
        # synchronously here, preventing background recompilation during the date loop
        # (which would otherwise slow down day 2 inference by 3-5x).
        _d_warmup = torch.zeros(5000, nchannel, depth, width, height,
                                device=device, dtype=torch.float32)
        for _n in [1000, 2000, 5000]:
            with autocast('cuda'):
                model(_d_warmup[:_n])
        del _d_warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[Rank {rank}] torch.compile warmup complete.")

        if world_size > 1:
            # Allocate the output buffer once and keep it alive for the entire date loop.
            # NCCL registers this memory address on the first all_reduce call and reuses
            # the registration on every subsequent call.  Deleting and re-allocating the
            # buffer each day causes NCCL proxy threads to de-register / re-register in
            # the background, which competes with GPU compute during day-2 inference.
            out_persistent = torch.zeros(len(lat_index), len(lon_index), device=device)
            # Perform multiple warmup all_reduces with realistic PM2.5-like data to
            # exhaust NCCL's auto-tuning phase before the date loop starts.
            # Values mix -9999 sentinels (fill pixels) with small positive values,
            # matching the actual day-1 output that previously triggered a new round
            # of NCCL benchmarking.  20 iterations cover the full tuning budget even
            # for large NVLink all_reduce with Ring/Simple protocol.
            _nlon = len(lon_index)
            for _w in range(20):
                out_persistent.fill_(-9999.0)
                # ~5% of pixels get a realistic positive PM2.5 value
                out_persistent[:max(1, len(lat_index) // 20), :max(1, _nlon // 20)] = float(_w % 50 + 1)
                dist.all_reduce(out_persistent, op=dist.ReduceOp.MAX)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            out_persistent.zero_()                    # reset to fill value for actual use
            print(f"[Rank {rank}] NCCL all_reduce warmup complete.")

        for date in int_AllDates:
            # Your original CNN3D_train logic goes here...
            YYYY,MM,DD = getGrg_YYYY_MM_DD(date)
            print(f"Processing date: {YYYY}-{MM}-{DD}")
            t0 = time.perf_counter()
            output = np.full((len(lat_index),len(lon_index)),-9999.0,dtype=np.float32)

            ## Load Map Data — reuse cached days when processing consecutive dates
            expected_prev_date = get_previous_date_YYYY_MM_DD(date, 1)  # yesterday
            if cached_map_data is not None and cached_last_date == expected_prev_date:
                # Shift in-place (no extra allocation): [0..depth-2] ← [1..depth-1], load slot depth-1
                # Left-shift is safe in-place: source slots are always ahead of destination in memory
                cached_map_data[:, :depth-1, :, :] = cached_map_data[:, 1:, :, :]
                cached_map_data[:, depth-1, :, :] = load_map_data(total_channel_names, YYYY, MM, DD, rank)
                temp_map_data = cached_map_data
            else:
                # Non-consecutive date or first date — load all days fresh
                temp_map_data = np.zeros((len(total_channel_names), depth, len(SATLAT), len(SATLON)), dtype=np.float32)
                def _load_day(iday):
                    temp_date = get_previous_date_YYYY_MM_DD(date, iday)
                    iYYYY, iMM, iDD = getGrg_YYYY_MM_DD(temp_date)
                    return depth - iday - 1, load_map_data(total_channel_names, iYYYY, iMM, iDD, rank)
                with _TPE(max_workers=depth) as ex:
                    for slot, data in ex.map(_load_day, range(depth)):
                        temp_map_data[:, slot, :, :] = data
            cached_map_data = temp_map_data
            cached_last_date = date
            t1 = time.perf_counter()

            ## Convert to 3D CNN reading and predict
            _diag_t_prep = 0.0   # CPU: patch extraction + normalize + NaN-fill
            _diag_t_gpu  = 0.0   # GPU: forward pass + H2D/D2H transfers
            _diag_rows   = 0
            _diag_pixels = 0
            _diag_nans   = 0
            for ix in range(len(lat_index)//world_size):
                ix = ix*world_size + rank
                land_index = np.where((landtype[ix,:] != 0) & (~np.isnan(temp_map_data[AOD_index, -1, lat_index[ix], lon_index])))

                if ix % max(1, len(lat_index) // 10) == 0:
                    print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' ,
                          'Time elapsed: {:.2f}s.'.format(time.perf_counter() - t1),
                          f'| prep={_diag_t_prep:.1f}s gpu={_diag_t_gpu:.1f}s rows={_diag_rows} pixels={_diag_pixels} nans={_diag_nans}')
                if len(land_index[0]) > 0:
                    _tc0 = time.perf_counter()
                    # Vectorized patch extraction — no Python loop over pixels
                    lat_start = int(lat_index[ix] - half_width)
                    lat_end   = int(lat_index[ix] + half_width + 1)
                    lon_coords  = lon_index[land_index[0]]                            # (N,)
                    lon_offsets = np.arange(-half_height, half_height + 1)            # (width,)
                    lon_indices = lon_coords[:, None] + lon_offsets[None, :]          # (N, width)
                    # lat_slice: (nchannel, depth, width, NLON)
                    # indexed by lon_indices (N, width) → (nchannel, depth, width, N, width)
                    # transpose → (N, nchannel, depth, width, width)
                    lat_slice  = temp_map_data[:, :, lat_start:lat_end, :]
                    temp_input = np.ascontiguousarray(
                        np.transpose(lat_slice[:, :, :, lon_indices], (3, 0, 1, 2, 4)),
                        dtype=np.float32
                    )

                    temp_input -= train_mean
                    temp_input /= train_std
                    center_values = temp_input[:, :, depth-1, half_width, half_height]
                    # 3. Broadcast center_vals to full patch shape: (N, nchannel, depth, width, width)
                    center_full = np.broadcast_to(center_values[:, :, None, None, None],temp_input.shape)
                    nan_index = np.where(np.isnan(temp_input))
                    temp_input[nan_index] = center_full[nan_index]
                    _tc1 = time.perf_counter()
                    _diag_t_prep += _tc1 - _tc0
                    _diag_rows   += 1
                    _diag_pixels += len(land_index[0])
                    _diag_nans   += len(nan_index[0])

                    t_input = torch.from_numpy(temp_input)
                    outputs = []
                    for start in range(0, len(t_input), INFER_BATCH_SIZE):
                        batch = t_input[start:start + INFER_BATCH_SIZE].to(device, non_blocking=True)
                        with autocast('cuda'):
                            outputs.append(model(batch).squeeze(1).cpu().numpy())
                    output[ix, land_index[0]] = np.concatenate(outputs)
                    _tc2 = time.perf_counter()
                    _diag_t_gpu += _tc2 - _tc1
                    # Log the very first batch latency per date to pinpoint when Day-2
                    # slowness starts (synchronize first to get an accurate wall-clock).
                    if _diag_rows == 1:
                        torch.cuda.synchronize()
                        _first_batch_s = _tc2 - _tc1
                        print(f"[DIAG] {YYYY}-{MM}-{DD} first-row latency: {_first_batch_s:.3f}s ({len(land_index[0])} px)")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            if world_size > 1:
                out_persistent.copy_(torch.from_numpy(output))  # CPU → GPU into NCCL-registered buffer
                dist.all_reduce(out_persistent, op=dist.ReduceOp.MAX)
                output = out_persistent.cpu().numpy().copy()    # .copy() so next date's overwrite is safe
                # Fully quiesce NCCL before the next date.
                # cudaDeviceSynchronize alone cannot stop NCCL proxy threads from
                # enqueueing new copy-engine work after it returns.  A dist.barrier()
                # is a collective NCCL fence: it cannot start until all prior NCCL
                # operations (including proxy-thread-scheduled work) have completed.
                dist.barrier()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            t3 = time.perf_counter()
            ## Save the output
            if rank == 0:
                output = map_data_final_output(output,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD, softplus_output)
                save_Estimation_Map(mapdata = output, outdir = data_recording_outdir,
            file_target='Map_Estimation', typeName=typeName, Area=Area, YYYY=YYYY, MM=MM, DD=DD, nchannel=nchannel,
            width=width, height=height, depth=depth
            )
            t4 = time.perf_counter()
            print(f'[TIMING] {YYYY}-{MM}-{DD} | load={t1-t0:.2f}s | infer={t2-t1:.2f}s | reduce={t3-t2:.2f}s | save={t4-t3:.2f}s | total={t4-t0:.2f}s')
    if world_size > 1:
        dist.barrier()
        destroy_process_group()
    return

def map_data_final_output(Validation_Prediction,
                         bias,normalize_bias,normalize_species,absolute_species,log_species,
                         true_mean, true_std, YYYY, MM, DD, softplus_output=False):
    lat_index, lon_index = get_extent_index(Extent)
    if bias == True:
        GeoSpecies = load_map_data('tSATPM25', YYYY, MM, DD, rank=0)
        validation_geophysical_species = GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
        final_data = Validation_Prediction + validation_geophysical_species
    elif normalize_bias == True:
        GeoSpecies = load_map_data('tSATPM25', YYYY, MM, DD, rank=0)
        validation_geophysical_species = GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
        final_data = Validation_Prediction * true_std + true_mean + validation_geophysical_species
    elif normalize_species == True:
        final_data = Validation_Prediction * true_std + true_mean
    elif absolute_species == True:
        final_data = Validation_Prediction
    elif log_species == True:
        final_data = np.exp(Validation_Prediction) - 1
    if softplus_output:
        valid_mask = final_data > -9000.0   # sentinel is -9999; exclude no-data pixels
        final_data[valid_mask] = np.logaddexp(0, final_data[valid_mask])
    return final_data
import os
import sys
import time
import h5py
import math
import shutil
import logging
import multiprocessing

import cv2                   as cv
import zarr                  as zr
import numpy                 as np
import tifffile              as tiff
import matplotlib.pyplot     as plt
import xml.etree.ElementTree as ET

from tqdm                   import tqdm
from dask                   import delayed
from pathlib                import Path
from suite2p.io             import combined
from suite2p.io.h5          import h5py_to_binary
from suite2p.io.tiff        import tiff_to_binary
from suite2p.run_s2p        import run_s2p, default_ops
from suite2p.extraction     import create_masks_and_extract, enhanced_mean_image, preprocess, oasis
from suite2p.classification import classify, builtin_classfile
from matplotlib.colors      import LogNorm
from distributed            import Client, as_completed
from scipy.ndimage          import maximum_filter, binary_dilation
from skimage.morphology import binary_dilation as bin_d
from skimage.morphology import binary_erosion as bin_e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#==============================| Parameters |=================================#
    
parent_dir = '/path/to/parent/directory'

skipping_params = {
    'skip_h5_convert': False,
    'skip_registration': False,
    'skip_tiffs_save': False,
}
convert_params = {
    'proc_channels': [0],
    'chunk_size': 10
}
preproc_params = {
    'steps': [10000, 2500, 1000, 500, 250, 100],
    'threshs': [1,2,3]
}
topseg_params = {
    'min_size': 6
}
ops_params = {
    'fish_tau': 0.2,
    'diameter': 4,
    'neuropil_extract': False
}

#================================| Variables |================================#

client = None

stepSizeUM = None
fps = None
Lz = None
Ly = None
Lx = None
Lt = None

volumes = None

#================================| Functions |================================#

def parse_metadata(metadata_path):
    global Lz, Lt, Ly, Lx, fps, stepSizeUM
    try:
        tree = ET.parse(metadata_path)
        root = tree.getroot()

        Lz = int(root.find(".//ZStage").attrib["steps"]) + 1
        Lt = int(root.find(".//Timelapse").attrib["timepoints"])
        Lx = int(root.find(".//LSM").attrib["pixelX"])
        Ly = int(root.find(".//LSM").attrib["pixelY"])
        n_channels = int(root.find(".//LSM").attrib.get("channels", "1"))
        fps = float(root.find(".//LSM").attrib["frameRate"])
        stepSizeUM = int(root.find(".//ZStage").attrib["stepSizeUM"])

        print(f"Metadata parsed: Lx={Lx}, Ly={Ly}, Lz={Lz}, "
              f"Lt={Lt}, n_channels={n_channels}")
        return n_channels, Lx, Ly, Lz, Lt
    except Exception as e:
        raise ValueError(f"Error parsing metadata file {metadata_path}: {e}")


def load_raw_chunk(file_path, t_idx, dtype, shape, channel):
    n_channels, Lx, Ly, Lz, Lt = shape
    single_channel_size = Lx * Ly * Lz 
    single_time_size = single_channel_size * n_channels

    if channel >= n_channels:
        sys.exit(1)

    offset = (t_idx * single_time_size + channel * single_channel_size) * np.dtype(dtype).itemsize

    with open(file_path, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=single_channel_size)

    data = data.reshape((Lz, Ly, Lx))   
    return data[:-1, :, :] # Removes flyback slice


def process_single_volume(raw_path, h5_path, t_idx, dtype, shape, channel):
    volume_data = load_raw_chunk(raw_path, t_idx, dtype, shape, channel)

    with h5py.File(h5_path, 'w') as h5_file:
        dset = h5_file.create_dataset('volume', data=volume_data, dtype=dtype)

def process_time_point(t_idx, raw_path, h5_folder, dtype, shape, channels_to_process):
    for channel in channels_to_process:
        h5_path = h5_folder / f"timepoint_{t_idx:04d}_channel_{channel}.h5"
        process_single_volume(raw_path, h5_path, t_idx, dtype, shape, channel)


def register(unreg_dir):
    ifolder = os.path.join(parent_dir, unreg_dir)
    ofolder = os.path.join(parent_dir, 'registration_output')
    
    fish_tau: float = ops_params['fish_tau']

    db = {
        'data_path': [ifolder],
        'save_path0': ofolder,
        'delete_bin': False,
        'h5py_key': ['volume'],
        'input_format': 'h5',

        'nplanes': Lz,
        'nchannels': 1,
        'functional_chan': 1,
        'tau': fish_tau,
        'fs': fps / (Lz + 1),

        'do_registration': True,
        'roidetect': False,

        'nimg_init': 100,
        'batch_size': 5000,
        'smooth_sigma': 1.5,
        'smooth_sigma_time': 2,
        'pad_fft':True,
        
        'nonrigid': True,
        'block_size': [128, 128],

        'sparse_mode': False,
        'spatial_scale': 1,
        'threshold_scaling': 0.1,
        'spatial_hp_detect': 10,
        'max_overlap': 1,
        'high_pass': 300 * (fps / (Lz + 1)),
        'smooth_masks': False,

        'neuropil_extract': False,
        'allow_overlap': True,

        'baseline': 'constant_percentile',
        'prctile_baseline': 10,
      
        'pre_load': True,

        'lenZ': stepSizeUM,
        'lenY': Ly,
        'lenX': Lx,
    }

    ops = default_ops()
    ops.update(db)
    ops['fast_disk'] = db['save_path0']
    ops['rerun_pipeline'] = False
    ops['delete_bin'] = False

    os.makedirs(db['save_path0'], exist_ok=True)

    try:
        logging.info("Starting Suite2p registration pipeline (parallel per-plane)...")
        start_time = time.time()
      
        ops = run_s2p(ops=ops, db={})

        logging.info("All planes registered and combined.")
        logging.info(f"Total pipeline runtime: {time.time() - start_time:.2f} seconds")

    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted. Saving partial results...")
        if 'ops' in locals():
            np.save(os.path.join(ops['save_path0'], 'partial_ops.npy'), ops)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during Suite2p pipeline: {e}")
        sys.exit(1)


def load_mempaps(suite2p_dir):
    sorted_dirs = sorted([
        d for d in os.listdir(suite2p_dir)
        if d.startswith('plane') and os.path.isdir(os.path.join(suite2p_dir, d))
    ], key=lambda x: int(x.replace('plane','')))
    plane_dirs = []
    for i in range(len(sorted_dirs)):
        plane_dirs.append(os.path.join(suite2p_dir, sorted_dirs[i]))


    memmaps = []
    Ly, Lx, nframes = None, None, None

    for plane_dir in plane_dirs:
        ops_path = os.path.join(plane_dir, 'ops.npy')
        if not os.path.exists(ops_path):
            logging.warning(f"Missing ops.npy in {plane_dir}, skipping.")
            continue

        ops = np.load(ops_path, allow_pickle=True).item()
        reg_file = ops.get('reg_file')

        if not reg_file or not os.path.exists(reg_file):
            logging.warning(f"Missing or invalid reg_file in {plane_dir}, skipping.")
            continue

        Ly, Lx = ops['Ly'], ops['Lx']
        data = np.memmap(reg_file, dtype=np.int16, mode='r').reshape(-1, Ly, Lx)

        if nframes is None:
            nframes = data.shape[0]
        elif data.shape[0] != nframes:
            print("ValueError: Frame mismatch across planes.")
            sys.exit(1)

        memmaps.append(data)

    if not memmaps:
        raise RuntimeError("No valid planes found.")
    
    return memmaps, (Ly, Lx), nframes


def write_single_volume(t, memmap_paths, shape, output_dir):
    Ly, Lx = shape
    vol = np.stack([
        np.memmap(mem_path, dtype=np.int16, mode='r').reshape(-1, Ly, Lx)[t]
        for mem_path in memmap_paths
    ], axis=0)  # shape: (z, y, x)

    tif_path = os.path.join(output_dir, f'vol_{t:05d}.tif')
    tiff.imwrite(tif_path, vol)


@delayed
def segment(path, plane_num: int, mask_cell: np.array, min_size: int):
    data = np.load(os.path.join(path, 'mean.npy'))
   
    def stratify(data: np.ndarray, mask: np.array, min_size: int):
        r = 2
        pad = np.pad(data, pad_width=r, mode='constant', constant_values=0)
        cells = np.zeros(pad.shape, dtype=np.uint16)
        bin_mask = np.zeros(cells.shape, dtype=bool)
        bin_mask[r:-r, r:-r] = True
        bin_mask[pad == 0] = False
        peak_mask = bin_mask.copy()
        count = 0

        for i in range(np.max(data), np.min(data), -1):
            stratum = (pad == i).astype(np.uint8)
            stratum = stratum * peak_mask

            if np.sum(stratum) > 0:
                labels = cv.connectedComponents(stratum)[1]
                
                values = range(1, np.max(labels)+1)
                avgs = {
                    label: data[labels[r:-r, r:-r] == label].mean()
                    for label in values
                }
                ordered = sorted(values, key=lambda l: avgs[l], reverse=True)
                
                for j in ordered:
                    cy, cx = np.where((peak_mask * labels) == j)
                    if len(cx) > 0 or len(cy) > 0:
                        my = int(np.median(cy))
                        mx = int(np.median(cx))
                    
                        cell_mask = np.zeros(pad.shape, dtype=bool)
                        cell_mask[my-r : my+r+1, mx-r : mx+r+1] = mask
                        cell_mask = cell_mask * bin_mask
                        dilated = binary_dilation(cell_mask)
                    
                        if np.sum(cell_mask) >= min_size:
                            cells[cell_mask == True] = count + 1
                            bin_mask[cell_mask == True] = False
                            peak_mask[dilated == True] = False
                            count += 1

        return cells[r:-r, r:-r]

    rois = stratify(data, mask_cell, min_size)
    np.save(os.path.join(path, 'temp_rois.npy'), rois)
    return len(np.unique(rois)) - 1, plane_num


def calc_compact(mask):
    area = np.count_nonzero(mask)
    perimeter_mask = binary_dilation(mask) ^ mask
    perimeter = np.count_nonzero(perimeter_mask)
    
    if area == 0:
        return np.inf
    return (perimeter ** 2) / (4 * np.pi * area)


@delayed
def build_stat(path, min_size: int, frame: int):
    data = np.load(os.path.join(path, 'temp_rois.npy'))
    stat = []
    
    for i in np.unique(data):
        if i != 0:
            mask = np.zeros(data.shape, dtype=bool)
            mask[data == i] = 1

            ypix, xpix = np.nonzero(mask)
            xpix = xpix.astype(np.int64)
            ypix = ypix.astype(np.int64)
            npix = len(xpix)
            
            lam    = mask[ypix, xpix].astype(np.float32)
            med    = np.array([np.median(xpix), np.median(ypix)], dtype=np.float32)
            
            xrad = int(math.ceil((np.max(xpix) - np.min(xpix)) / 2))
            yrad = int(math.ceil((np.max(ypix) - np.min(ypix)) / 2))
            radius = xrad if xrad > yrad else yrad
            compact = calc_compact(mask)
            
            aspect_ratio = 2 * yrad / (0.01 + yrad + xrad)
            
            if npix >= min_size:
                stat.append({
                    'xpix': xpix, 
                    'ypix': ypix, 
                    'npix': npix, 
                    'lam': lam, 
                    'med': med, 
                    'radius': radius, 
                    'compact': compact,
                    'footprint': 1.0,
                    'aspect_ratio': aspect_ratio
                })
            
    with open(os.path.join(path, 'stat.npy'), 'wb') as f:   
        np.save(f, np.array(stat))
        f.flush()
        os.fsync(f.fileno())

    return frame


@delayed
def build_ops(path, dims: (int,int,int,int), frame: int, fps: float, stepSizeUM: float, fish_tau: float, diameter: int, neuropil_extract: bool, defaults: dict):
    plane = f'plane{frame}'
    data = np.load(os.path.join(path, 'suite2p', plane, 'mean.npy'))
    
    ipath = path
    opath = os.path.join(path, 'suite2p', plane)

    x = dims[3]
    y = dims[2]
    z = dims[0]
    t = dims[1]
    
    ops = default_ops()
    ops.update(defaults)

    db = {
        # Pathing
        'data_path':  [str(ipath)],
        'save_path0': str(opath),   
        'fast_disk':  str(opath),
        'reg_file':   os.path.join(opath, 'data.bin'),
        
        #Main params
        'nplanes': z,
        'nframes': t,
        'frame': frame,
        'fs': fps / (z + 1),
        'Lx': x,
        'Ly': y,
        'xrange': np.array([0, x]),
        'yrange': np.array([0, y]),
        'diameter': diameter,
            
        #Params for later analysis not utilized by Suite2p
        'lenX':x,
        'lenY':y,
        'lenZ':stepSizeUM,
        
        # Temp
        'meanImg': data
    }
    ops.update(db)

    with open(os.path.join(opath, 'ops.npy'), 'wb') as f:   
        np.save(f, ops)
        f.flush()
        os.fsync(f.fileno())
    return frame


def format_time(delta_t: float):
    secs = delta_t % 60
    mins = int((delta_t % 3600) // 60)
    hour = int(delta_t // 3600)
    
    if hour > 0:
        return f"{hour}h {mins}m {secs:.2f}s"
    if mins > 0:
        return f"{mins}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


if __name__ == "__main__":
    t0 = time.time()
    
    #==| Initialize Dask client |=================================================================
    multiprocessing.set_start_method('fork', force=True)

    client = Client("tcp://203.0.113.10:8080")
    print("Dask client connected:", client)
    workers = client.scheduler_info()['workers']
    # filtered_workers = [w for w in workers if target_host in w]
    filtered_workers = workers
    print("Number of Filtered workers:", len(filtered_workers))

    #==| Convert raw files into h5 files |========================================================
    try:
        folder_path = parent_dir
        metadata_file = Path(folder_path) / "Experiment.xml"

        try:
            n_channels, Lx, Ly, Lz, Lt = parse_metadata(metadata_file)
        except ValueError as e:
            print(e)
            sys.exit(1)

        dtype = np.uint16  # Update based on your .raw file's data type
        data_shape = (n_channels, Lx, Ly, Lz, Lt)
        print('Raw data shape is:', n_channels, Lx, Ly, Lz, Lt)

        # print(f"Available channels: {', '.join(map(str, range(n_channels)))}")
        channels_to_process = convert_params['proc_channels']
        print("Selected channels:", channels_to_process)
        # channels_to_process = [int(c.strip()) for c in channels_to_process if int(c.strip()) < n_channels]

        if not channels_to_process:
            print("No valid channels selected. Exiting...")
            sys.exit(1)

        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist.")

        run_conversion = (skipping_params['skip_h5_convert'] == False)
        if run_conversion:
            raw_files = list(folder.glob("Image*.raw"))
            if not raw_files:
                print("No .raw files found starting with 'Image'.")
            else:
                chunk_size = convert_params['chunk_size']
                for raw_file in raw_files:
                    print(f"Processing {raw_file.name}...")
                    h5_folder = raw_file.parent / "h5file"
                    h5_folder.mkdir(parents=True, exist_ok=True)

                    for chunk_start in range(0, Lt, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, Lt)
                        print(f"Processing time points {chunk_start} to {chunk_end - 1} with multiprocessing...")

                        args = [(t_idx, raw_file, h5_folder, dtype, data_shape, channels_to_process) for t_idx in range(chunk_start, chunk_end)]
                        _t_idx, _raw_file, _h5_folder, _dtype, _shape, _channels_to_process = zip(*args)
    
                        futures = client.map(process_time_point, _t_idx, _raw_file, _h5_folder, _dtype, _shape, _channels_to_process, workers=filtered_workers)
                        client.gather(futures)

                    print("Chunked processing with multiprocessing complete.")
                    
                print("All files converted. HDF5 files are saved in 'h5clean' subfolders.")
        else:
            print("HDF5 files already converted. Skipping step...")

        Lz = Lz - 1
        print('Data shape is:', n_channels, Lx, Ly, Lz, Lt)
    except Exception as e:
        print(f'RuntimeError: Something went wrong with raw-to-h5 conversion: {e}.')
        sys.exit(1)

    #==| Run Suite2p Registration |============================================#
    try:
        print('Registering data...')
        run_registration = (skipping_params['skip_registration'] == False)
        if run_registration:
            unreg_dir = 'h5files'
            register(unreg_dir)
        else:
            print('Registration already calculated. Skipping step...')

    except Exception as e:
        print(f'RuntimeError: Something went wrong with Suite2p registration: {e}.')
        sys.exit(1)

    #==| Save Tiffs |==========================================================#
    try:
        print('Saving tiffs...')
        run_tiffs_save = (skipping_params['skip_tiffs_save'] == False)
        if run_tiffs_save:
            ofolder = os.path.join(parent_dir, 'registration_output')
            suite2p_dir = os.path.join(ofolder, 'suite2p')
            registered_tif_dir = os.path.join(suite2p_dir, 'registered_tif')
            os.makedirs(registered_tif_dir, exist_ok=True)

            logging.info("Loading memory maps from Suite2p planes...")
            memmaps, (Ly, Lx), nframes = load_mempaps(suite2p_dir)
            nplanes = len(memmaps)
            logging.info(f"Found {nplanes} planes, {nframes} frames, shape: ({Ly}, {Lx})")

            memmap_paths = [m.filename for m in memmaps]

            logging.info(f"Starting multiprocessing to write TIFFs with {len(filtered_workers)} workers...")

            args = [(t, memmap_paths, (Ly, Lx), registered_tif_dir) for t in range(nframes)]
            _t, _memmap_paths, _dims, _reg_tif_dir = zip(*args)

            futures = client.map(write_single_volume, _t, _memmap_paths, _dims, _reg_tif_dir, workers=filtered_workers)
            progress = tqdm(total=len(futures), desc="Writing TIFFs")

            for job in as_completed(futures):
                progress.update(1)
            progress.close()
            logging.info(f"Saved {nframes} TIFFs to {registered_tif_dir}")

            print('Reorganizing directories...')
            shutil.move(registered_tif_dir, os.path.join(parent_dir, 'registered_tif'))
            shutil.move(suite2p_dir, os.path.join(parent_dir, 'suite2p'))

        else:
            print('Registered TIFF folder already saved. Skipping step...')

    except Exception as e:
        print(f'RuntimeError: Something went wrong with saving tiffs: {e}.')
        sys.exit(1)

    #==| Run Preprocessing |===================================================#
    try:
        print('Processing tiffs...')
        tif_dir = os.path.join(parent_dir, 'registered_tif')
        out_dir = 'registered_tif' if preproc_params['overwrite_tif'] else 'clean_tif'
        os.makedirs(out_dir, exist_ok=True)
        tiffs = sorted(os.listdir(tif_dir))

        roll_sum = np.zeros((Lz,Ly,Lx), dtype=np.float32)
        for i in tqdm(range(len(tiffs)), desc='Averaging registered tiffs'):
            data = tiff.imread(os.path.join(tif_dir, tiffs[i]))
            roll_sum += data

        roll_sum /= len(tiffs)
        vmin = roll_sum.min()
        vmax = roll_sum.max()
        roll_sum -= vmin
        roll_sum /= (vmax - vmin)
        roll_sum *= 65535
        roll_sum = np.rint(roll_sum).astype(np.uint16)

        tiff.imwrite(os.path.join(parent_dir, 'reg_tiff_avg.tif'), roll_sum, imagej=True, metadata={'axes':'ZYX'})

        mean = roll_sum
        del roll_sum, vmin, vmax
        ref = mean[10,:,:]

        threshs = np.zeros((50, ref.shape[0], ref.shape[1]), dtype=np.uint8)
        for i in range(50):
            new = (ref > i*50).astype(np.uint8)
            new = bin_e(new)
            new = bin_d(new)

            threshs[i,:,:] = new * 255

        tiff.imwrite(os.path.join(parent_dir, 'threshs.tif'), threshs, imagej=True, metadata={'axes':'ZYX'})

        thresh = int(input("Select mask number from 'threshs.tif' to mask out background: ")) - 1
        mask = (mean > thresh*50)
        for z in range(mean.shape[0]):
            mask[z,:,:] = bin_e(mask[z,:,:])
            mask[z,:,:] = bin_d(mask[z,:,:])
        tiff.imwrite(os.path.join(parent_dir, 'mask.tif'), mask.astype(np.uint8)*255, imagej=True, metadata={'axes':'ZYX'})
        del mean

        reg_dir = os.path.join(parent_dir, 'registered_tif')
        out_dir = os.path.join(parent_dir, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        mean = np.zeros(mask.shape, dtype=np.float32)
        volumes = np.zeros((mask.shape[0], len(tiffs), mask.shape[1], mask.shape[2]), dtype=np.int16)
        for i in tqdm(range(len(tiffs)), desc='Scanning tifs'):
            vol = tiff.imread(os.path.join(reg_dir, tiffs[i]))
            vol *= mask
            mean += vol
            volumes[:,i,:,:] = vol

        mean /= len(tiffs)

        vmin = mean.min()
        vmax = mean.max()
        mean -= vmin
        mean /= (vmax - vmin)
        mean *= 65535
        mean = np.rint(mean).astype(np.uint16)

        tiff.imwrite(os.path.join(parent_dir, 'clean_tiff_avg.tif'), mean, imagej=True, metadata={'axes':'ZYX'})
        del tiffs, mean, vmin, vmax

    except Exception as e:
        print(f'RuntimeError: Something went wrong with preprocessing: {e}.')
        sys.exit(1)

    #==| Run TopSeg v1.0 |=====================================================#
    try:
        ref_avg = tiff.imread(os.path.join(parent_dir, 'clean_tiff_avg.tif'))
        for i in range(ref_avg.shape[0]):
            plane_dir = os.path.join(parent_dir, 'suite2p', f'plane{i}')
            np.save(os.path.join(plane_dir, 'mean.npy'), ref_avg[i,:,:])
        del ref_avg

        topseg_ops_defaults = {
            'input_format':     'binary',
            'data_dtype':       'int16',
            'use_dask':         True,    
            'delete_bin':       False,
            'rerun_pipeline':   False,
            'do_registration':  False,
            'roidetect':        False,
            'signal_extract':   True,
            'neuropil_extract': True,
            'allow_overlap':    True,
            'combined':         True,    
            'nchannels':        1,
            'functional_chan':  1,
            'baseline':         'constant_percentile',
            'prctile_baseline': 1,
        }
        
        mask_5x5 = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        print('Running topseg...')
        filepaths = [os.path.join(parent_dir, 'suite2p', f'plane{i}') for i in range(Lz)]
        
        tasks = [segment(filepaths[i], i, mask_5x5, topseg_params['min_size']) for i in range(Lz)]
        segmented = client.compute(tasks, workers=filtered_workers)

        n = 1
        count = 0
        for job in as_completed(segmented):
            print(f'Found {job.result()[0]} cells in plane{job.result()[1]} ({n}/{Lz})')
            count = count + job.result()[0]
            n = n + 1

        print(f'Identified {count} total cells')
    except Exception as e:
        print(f'RuntimeError: Something went wrong with TopSeg v1.0 segmentation: {e}.')

    #==| Building stat.npy |=================================================#
    try:
        print('Building stat.npy files...')
        tasks = [build_stat(filepaths[i], topseg_params['min_size'], i) for i in range(Lz)]
        stat_files = client.compute(tasks, workers=filtered_workers)
        
        n = 1
        for job in as_completed(stat_files):
            print(f'Created stat.npy for plane{job.result()} ({n}/{Lz})')
            n = n + 1
    except Exception as e:
        print(f'RuntimeError: Something went wrong with building stat.npy files: {e}.')

    #==| Building ops.npy |==================================================#    
    try:
        print('Building ops.npy files...')
        tasks = [build_ops(parent_dir, (Lz, Lt, Ly, Lx), i, fps, stepSizeUM, ops_params['fish_tau'], ops_params['diameter'], ops_params['neuropil_extract'], topseg_ops_defaults) for i in range(Lz)]
        ops_files = client.compute(tasks, workers=filtered_workers)
        
        n = 1
        for job in as_completed(ops_files):
            print(f'Created ops.npy for plane{job.result()} ({n}/{Lz})')
            n = n + 1
        print('Segmentation completed')
    except Exception as e:
        print(f'RuntimeError: Something went wrong with building ops.npy files: {e}.')
 
    #==| Run Suite2p Spike Deconvolution |==================================#s
    try:
        tif_dir = 'registered_tif'
        ifolder = os.path.join(parent_dir, tif_dir)
        ofolder = parent_dir
        
        db = {
            # Pathing
            'data_path':   [ifolder],
            'save_path0':  ofolder,
            'fast_disk':   ofolder,
            'reg_file':    ofolder,
            'ops_path':    ofolder,
            'data_dtype':  'int16',
            'input_format': 'tiff',
        
            # Processes
            'use_dask':         True,    
            'delete_bin':       False,
            'combined':         True,
            'do_registration':  False,
            'spikedetect':      True,
            'neuropil_extract': ops_params['neuropil_extract'],
            'allow_overlap':    True,
            
            #Main params
            'nchannels': 1,
            'nframes': Lt,
            'functional_chan': 1,
            'fs': fps / (Lz + 1),
            'Lx': Lx,
            'Ly': Ly,
            'xrange': np.array([0, Lx]),
            'yrange': np.array([0, Ly]),
            'diameter': ops_params['diameter'],
            'batch_size': 20000,
            'tau': ops_params['fish_tau'],
        
            #Spike deconvolution params
            'baseline': 'constant_percentile',
            'prctile_baseline': 10,
        
            # 'lenX': Lx,
            # 'lenY': Ly,
            # 'lenZ': stepSizeUM,
            # 'Vcorr': np.zeros((Ly, Lx), dtype=np.float32),
        }

        ops = default_ops()
        ops.update(db)
        
        save_path = db['save_path0']
        os.makedirs(save_path, exist_ok=True)

        print("Starting Suite2p processing...")
        step_start_time = time.time()

        for i in range(Lz):
            path = os.path.join(ofolder, 'suite2p', f'plane{i}')
            data = np.load(os.path.join(path, 'stat.npy'), allow_pickle=True)
            
            ops['save_path'] = path
            ops['reg_file']  = os.path.join(path, 'data.bin')
            ops['ops_path']  = os.path.join(path, 'ops.npy')
            ops['meanImg']   = np.mean(volumes[i,:,:,:], axis=0)
            ops['max_proj']  = np.max(volumes[i,:,:,:], axis=0)
     
            # Generate traces
            stat, F, Fneu, _, _ = create_masks_and_extract(ops=ops, stat=data)
     
            np.save(os.path.join(path, 'stat.npy'), stat)
            np.save(os.path.join(path, 'F.npy'), F)
            np.save(os.path.join(path, 'Fneu.npy'), Fneu)
      
            # Generate iscell masks
            iscell = classify(stat=stat, classfile=builtin_classfile)
     
            np.save(os.path.join(path, 'iscell.npy'), iscell)
    
            # Prepare traces for spike deconvolution
            dF = F.copy() - ops['neucoeff'] * Fneu
            dF = preprocess(
                F=dF, 
                baseline=ops['baseline'], 
                win_baseline=ops['win_baseline'], 
                sig_baseline=ops['sig_baseline'], 
                fs=ops['fs'],
                prctile_baseline=ops['prctile_baseline']
            )
  
            # Deconvolute spike traces
            spks = oasis(
                F=dF,
                batch_size=ops['batch_size'],
                tau=ops['tau'],
                fs=ops['fs']                
            )
      
            np.save(os.path.join(path, 'spks.npy'), spks)

            # Generate enhanced mean image for suite2p gui
            temp_ops = enhanced_mean_image(ops=ops)
            new_ops = np.load(os.path.join(path, 'ops.npy'), allow_pickle=True).item()
            for prop in ['meanImgE', 'aspect', 'spatscale_pix', 'xrange', 'yrange', 'Vcorr', 'do_registration', 'combined', 'Ly', 'Lx']:
                new_ops[prop] = temp_ops[prop]            
            np.save(os.path.join(path, 'ops.npy'), new_ops, allow_pickle=True)
            print(f'Created data for plane{i} ({i+1}/{Lz})')

        # stat, ops, F, Fneu, spks, _, _, _, _, _ = combined(os.path.join(ofolder, 'suite2p'), save=True)    
        stat, ops, F, Fneu, spks, _, _, _, _, _ = combined(os.path.join(ofolder, 'suite2p'), save=False)    
        os.makedirs(os.path.join(folder, 'suite2p', 'combined'), exist_ok=True)     
        np.save(os.path.join(ofolder, 'suite2p', 'combined', 'stat.npy'), stat)
        np.save(os.path.join(ofolder, 'suite2p', 'combined', 'F.npy'), F)
        np.save(os.path.join(ofolder, 'suite2p', 'combined', 'Fneu.npy'), Fneu)
        np.save(os.path.join(ofolder, 'suite2p', 'combined', 'spks.npy'), spks)
        np.save(os.path.join(ofolder, 'suite2p', 'combined', 'ops.npy'), ops)

        step_end_time = time.time()
        print(f"Suite2p processing completed successfully in {step_end_time - step_start_time:.2f} seconds.")

    except Exception as e:
        print(f'RuntimeError: Something went wrong with Suite2p spike deconvolution: {e}.')
        sys.exit(1)

    finally:
        client.close()
        print("Dask workers stopped.")
        print('Pipeline complete.')
        print(f'Total run time: {format_time(time.time() - t0)}')

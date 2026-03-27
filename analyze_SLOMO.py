import os
import sys
import h5py
import logging
import numpy as np
import numba as nb
import scipy as sp
import pandas as pd
import seaborn as sb
import sklearn as sk
import tifffile as tiff
import statsmodels as sm
import multiprocessing as mp
import matplotlib.pyplot as plt
import skimage.morphology as morph
import xml.etree.ElementTree as ET
import matplotlib.animation as anim
import sklearn.decomposition

from tqdm import tqdm
from pathlib import Path
from functools import partial
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

#=================================================================================================================#

def _check_path(path, new=False):
    if type(path) == str:
        path = Path(path)
    if (not new) and (not path.exists()):
        log.error(" Path '%s' does not exist", path)
        sys.exit(1)
    return path

def _find_file(path, pattern):
    path = _check_path(path)

    files = list(path.glob(pattern))
    if len(files) == 0:
        log.error("No '%s' file found in '%s'", pattern, path)
        sys.exit(1)
    if len(files) > 1:
        log.error("Found %s file(s) with the pattern '%s' in '%s'", len(files), pattern, path)
        sys.exit(1)

    file = files[0]
    return file

def _load_file(path, pattern, pickled=False):
    path = _check_path(path)
    file = _find_file(path, pattern)
    
    if file.suffix == '.npy':
        data = np.load(file, allow_pickle=pickled)
    elif file.suffix == '.tif':
        data = tiff.imread(file)
    elif file.suffix in ['.h5', '.hdf5']:
        data = h5py.File(file, 'r')
    else:
        log.error("Unrecognized file extension: '%s'", file.suffix)
        sys.exit(1)
    return data

def _load_fps(path):
    path = _check_path(path)
    fps_file = _find_file(path, '*frequency.txt')
    with open(fps_file, 'r') as file:
        Fps = float(file.readline().strip())
    return Fps

def _load_resolution(path):
    path = _check_path(path)
    file = _find_file(path, '*.xml')

    Ry, Rx = (0.406, .406)
    data = ET.parse(file)
    root = data.getroot()
    for info in root.findall('info'):
        if (Rz := info.get('z_step')) is not None:
            return float(Rz), Ry, Rx
    else:
        log.error("No 'z_step' tag found in '%s'", file.name)
        sys.exit(1)
    return

def _load_metadata(path):
    path = _check_path(path)
    Rz, Ry, Rx = _load_resolution(path)
    Rt = 1 / _load_fps(path)
    return Rz, Ry, Rx, Rt

def _ensure_activity_path(path):
    path = _check_path(path)
    if (path / 'voluseg').exists():
        path = path / 'voluseg'
    if path.name != 'voluseg':
        log.error('No voluseg directory found at %s', path)
        sys.exit(1)

    needed_files = ['cells0_clean.hdf5', 'volume0.hdf5']
    for file in needed_files:
        if path / file not in list(path.iterdir()):
            log.error("File '%s' not found in %s", file, path)
            sys.exit(1)
    return path

def _load_activity_data(path):
    path = _ensure_activity_path(path)
    cell_data = h5py.File(path / 'cells0_clean.hdf5', 'r')
    volume_data = h5py.File(path / 'volume0.hdf5', 'r')

    raw_traces = cell_data['cell_timeseries'][:].astype(np.float32)
    baseline = cell_data['cell_baseline'][:].astype(np.float32)
    cell_traces = (raw_traces - baseline) / _divisor(baseline)
    cell_traces = (cell_traces - cell_traces.mean()) / cell_traces.std()
    assert raw_traces.shape == cell_traces.shape == baseline.shape

    bmap = volume_data['volume_mean'][:].astype(np.float32)
    rois = cell_data['volume_id'][:].T.astype(np.int64)

    shape = (
        cell_traces.shape[0],    # Lc, cell count
        cell_traces.shape[1],    # Lt, timepoints
        bmap.shape[0],           # Lz, length Z
        bmap.shape[1],           # Ly, length Y
        bmap.shape[2]            # Lx  length X
    )
    return rois, cell_traces, bmap, shape


def _split_trials(drift):
    drft_mask = (drift != 0)
    move_mask = morph.remove_small_objects(drft_mask, min_size=10)
    wait_mask = ~move_mask
    puls_mask = wait_mask & (drift != 0)

    labels = morph.label(move_mask)
    sums = []
    for i in range(1, labels.max() + 1):
        sums.append(np.sum(labels == i))
    counts = []
    for i in np.unique(sums):
        counts.append((i, sum([sums[j] == i for j in range(len(sums))])))
    move_sort = sorted(counts, key=lambda count: count[1])
    move_length = move_sort[-1][0]
    move_num = move_sort[-1][1]

    labels = morph.label(wait_mask)
    sums = []
    for i in range(1, labels.max() + 1):
        sums.append(np.sum(labels == i))
    counts = []
    for i in np.unique(sums):
        counts.append((i, sum([sums[j] == i for j in range(len(sums))])))
    wait_sort = sorted(counts, key=lambda count: count[1])
    wait_length = wait_sort[-1][0]
    wait_num = wait_sort[-1][1]

    Ltt = wait_length + move_length
    Lt = len(drift)
    Ln = min(move_num, wait_num)

    starts = np.nonzero(np.diff(move_mask.astype(np.uint8)) == 1)[0] + 1
    _trials = []
    for i in range(len(starts)):
        if i == len(starts) - 1:
            if Lt - starts[i] >= Ltt:
                _trials.append(starts[i] + np.arange(Ltt))
            continue

        _Ltt = starts[i+1] - starts[i]
        if _Ltt != Ltt:
            continue
        
        _trials.append(starts[i] + np.arange(Ltt))

    Ln = len(_trials)
    trials = np.zeros((Ln, Ltt), np.int16)
    for n in range(Ln):
        trials[n,:] = _trials[n]

    pulses = puls_mask[trials]
    pulse = np.nonzero(pulses.sum(axis=0))[0]
    pulses[:,pulse] = True
    return trials, move_mask[trials], wait_mask[trials], pulses

def _decay(data, tau=2.00, width=16, inv=False):
    krnl = tau ** np.arange(width)
    krnl = np.exp(-np.arange(width) / tau)
    if inv:
        return np.convolve(data[::-1], krnl, mode='full')[:len(data)][::-1]
    return np.convolve(data, krnl, mode='full')[:len(data)]

def _divisor(arr, minimum=1, default_positive=True):
    default_sign = 1 if default_positive else -1
    signs = np.sign(arr)
    signs[signs == 0] = default_sign
    return signs * np.maximum(np.abs(arr), minimum)

def _draw_weights(rois, weights):
    fmap = np.zeros(rois.shape, np.float32)
    mask = rois > -1
    fmap[mask] = weights[rois[mask]]
    return fmap

def _draw_projections(weights, bmap, path, resolution, vmin=None, vmax=None, pmin=None, pmax=None, overlay=False):
    if pmin is not None:
        vmin = np.percentile(weights, pmin)
    else:
        vmin = vmin if vmin is not None else weights.min()
    if pmax is not None:
        vmax = np.percentile(weights, pmax)
    else:
        vmax = vmax if vmax is not None else weights.max()


    weights = np.repeat(weights, resolution[0]/resolution[1], axis=0)
    bmap = np.repeat(bmap, resolution[0]/resolution[1], axis=0)

    def subplot(regn, axis, cmap='gray'):
        if cmap == 'gray':
            plt.imshow(regn.max(axis=axis), vmin=np.percentile(bmap, 50), vmax=np.percentile(bmap, 99), cmap='gray')
            plt.ylabel('Ly')
        else:
            if overlay:
                plt.imshow(bmap.max(axis=axis), vmin=np.percentile(bmap, 50), vmax=np.percentile(bmap, 99), cmap='gray')
                plt.imshow(regn.max(axis=axis), vmin=vmin, vmax=vmax, cmap=RED_OVERLAY, alpha=0.7)
            else:
                plt.imshow(regn.max(axis=axis), vmin=vmin, vmax=vmax, cmap='inferno')
        if i == 0:
            plt.title('Transverse Plane')
        elif i == 1:
            plt.title('Sagittal Plane')
        else:
            plt.title('Coronal Plane')
            plt.xlabel('Lx')

    plt.figure()
    if overlay:
        for i in range(3):
            plt.subplot(3,1,i+1)
            subplot(weights, i, 'inferno')
    else:
        for i in range(3):
            plt.subplot(3,2,(i*2)+1)
            subplot(bmap, i)
            plt.subplot(3,2,(i*2)+2)
            subplot(weights, i, 'inferno')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return

RED_OVERLAY = LinearSegmentedColormap('red_overlay', {
    'red':   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    'alpha': ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})

#=================================================================================================================#

if __name__ == '__main__':

    pdir = Path('/path/to/parent/directory')
    ddirs = [
        Path('/path/to/data/directory/ND_fish1'),
        Path('/path/to/data/directory/ND_fish2'),
        Path('/path/to/data/directory/ND_fish3'),
        Path('/path/to/data/directory/HCD_fish1'),
        Path('/path/to/data/directory/HCD_fish2'),
        Path('/path/to/data/directory/HCD_fish3'),
    ]
    bdir = 'behDir'
    adir = 'analysis'
    gdir = 'graphs'

    params = {
        'cpus': 100,
        'bnum': 100,
        'bval': 5,
        'kval': 5,
        'clip': False,
        'norm': False,
        'invt': False
    }

    regions = [
        ((0,18),(300,725),(480,700)),
        ((0,15),(375,750),(400,800)),
        ((10,23),(300,800),(350,600)),

        ((6,30),(325,650),(750,1100)),
        ((0,16),(500,900),(400,550)),
        ((0,16),(350,900),(300,750))
    ]

    regns = []
    comps = []
    antis = []
    regns_dff  = []
    regns_dff2 = []
    regns_dff3 = []
    antis_dff = []

    assert pdir.exists()
    adir.mkdir(exist_ok=True)
    gdir.mkdir(exist_ok=True)

    #==============================================================================================================#

    for i in range(1,6):
        log.info('Analyzing dataset: %s', ddirs[i])

        rois, trcs, bmap, shape = _load_activity_data(ddirs[i])
        Lc, Lt, Lz, Ly, Lx = shape
        log.info('Data loaded (Lc, Lt, Lz, Ly, Lx): %s', shape)
        Rz, Ry, Rx, Rt = _load_metadata(ddirs[i])
        log.info('Metadata loaded (Rz, Ry, Rx, Rt): (%.3f, %.3f, %.3f, %.3f)', Rz, Ry, Rx, Rt)

        drft = _load_file(ddirs[i] / bdir, 'drift*')
        trials, move_mask, wait_mask, puls_mask = _split_trials(drft)
        Ln, Ltt = trials.shape
        log.info('Trials loaded (Ln, Ltt): (%s, %s)', Ln, Ltt)

        bhvr = _load_file(ddirs[i] / bdir, 'behavior*')
        gain = _load_file(ddirs[i] / bdir, 'gain*')
        vlct = _load_file(ddirs[i] / bdir, 'velocity*')
        swim = _load_file(ddirs[i] / bdir, 'swimbout*')
        log.info('Behavior data loaded')

        #old = ((0,47),(200,1000),(300,750))
        #new = ((0,16),(350,900),(300,750))
        #mask = np.zeros(bmap.shape, bool)
        #(z1,z2),(y1,y2),(x1,x2) = new
        #mask[z1:z2,y1:y2,x1:x2] = True

        #regn = np.load(pdir / f'regn{i}.npy')
        #print(len(regn))
        #temp = np.zeros(Lc, np.float32)
        #temp[regn] = 10.0
        #weights = _draw_weights(rois, temp)
        #weights[~mask] = weights.min()
        #_draw_projections(weights, bmap, pdir / f'checking{i}.pdf', resolution=(Rz,Ry,Rx), overlay=True)



        regn = np.load(pdir / f'regn{i}.npy')
        temp = np.zeros(Lc, np.float32)
        temp[regn] = 10.0
        (z1,z2),(y1,y2),(x1,x2) = regions[i]
        mask = np.zeros(bmap.shape, bool)
        mask[z1:z2,y1:y2,x1:x2] = True
        weights = _draw_weights(rois, temp)
        weights[~mask] = weights.min()
        #_draw_projections(weights, bmap, pdir / f'checking{i}.pdf', resolution=(Rz,Ry,Rx), overlay=True)
        
        weights = np.repeat(weights, Rz//Rx, axis=0)
        bmap = np.repeat(bmap, Rz//Rx, axis=0)

        plt.figure()
        plt.imshow(bmap.max(axis=0), vmin=np.percentile(bmap, 50), vmax=np.percentile(bmap, 99), cmap='gray')
        plt.imshow(weights.max(axis=0), cmap=RED_OVERLAY, alpha=0.8)
        plt.savefig(pdir / 'example_SLOMO1.pdf')
        plt.close()

        plt.figure()
        plt.imshow(bmap.max(axis=1), vmin=np.percentile(bmap, 50), vmax=np.percentile(bmap, 99), cmap='gray')
        plt.imshow(weights.max(axis=1), cmap=RED_OVERLAY, alpha=0.8)
        plt.savefig(pdir / 'example_SLOMO2.pdf')
        plt.close()

        plt.figure()
        plt.imshow(bmap.max(axis=2), vmin=np.percentile(bmap, 50), vmax=np.percentile(bmap, 99), cmap='gray')
        plt.imshow(weights.max(axis=2), cmap=RED_OVERLAY, alpha=0.8)
        plt.savefig(pdir / 'example_SLOMO3.pdf')
        plt.close()

        #break

        def _decay(data, path, inv=False):
            decay = _decay(data, inv=inv)
            plt.figure()
            plt.plot(data[:300], color='blue', alpha=0.4)
            plt.plot(decay[:300], color='red', alpha=0.4)
            plt.savefig(path)
            plt.close()
            return decay

        def _draw(corrs, path):
            plt.figure()
            plt.hist(corrs[:,0], bins=50)
            plt.axvline(x=0.0, color='red')
            plt.savefig(path)
            plt.close()

        move = move_mask.ravel().astype(np.uint8)
        wait = wait_mask.ravel().astype(np.uint8)
        puls = puls_mask.ravel().astype(np.uint8)
        Fpls = np.clip(puls * drft[trials.ravel()],  0, 1).astype(np.uint8)
        Bpls = np.clip(puls * drft[trials.ravel()], -1, 0).astype(np.uint8)
        Npls = puls * (drft == 0)[trials.ravel()].astype(np.uint8)
        drftD  = _decay(drft, width=Ltt)
        drftDi = _decay(drft, inv=True)
        bhvrD  = _decay(bhvr)
        bhvrDi = _decay(bhvr, inv=True)
        predDi = drftDi[trials.ravel()] * wait
        planDi = bhvrDi[trials.ravel()] * wait

        types = sorted(np.unique(drft[trials].sum(axis=1)))
        bkwd = np.where(drft[trials].sum(axis=1) == types[0])[0]
        nowd = np.where(drft[trials].sum(axis=1) == types[1])[0]
        frwd = np.where(drft[trials].sum(axis=1) == types[2])[0]
        log.info('Trial masks built')


#         log.info('Analzing antimotor activity')
#         corrs = np.zeros((Lc, 2), np.float32)
#         for c in range(Lc):
#             corrs[c,:] = sp.stats.spearmanr(trcs[c,:][trials.ravel()], _decay(-swim)[trials.ravel()])        
#         weights = _draw_weights(rois, corrs[:,0])
#         _draw_projections(weights, bmap, pdir / f'spmn_antimotorD_fmap{i}.pdf', resolution=(Rz,Ry,Rx), pmax=99.9, pmin=75)
#         _draw(corrs, pdir / f'spmn_antimotorD_hist{i}.pdf')

#         antimotor = np.intersect1d(np.where(corrs[:,0] > np.percentile(corrs[:,0], 99.9))[0], np.where(corrs[:,1] < 0.001)[0])
#         print('Lc* ->', len(antimotor))
        
#         temp = np.zeros(Lc, np.float32)
#         temp[antimotor] = 10.0
#         weights = _draw_weights(rois, temp)
#         _draw_projections(weights, bmap, pdir / f'spmn_antimotorD_selected{i}.pdf', resolution=(Rz,Ry,Rx))

#         temp = np.zeros(Lc, np.float32)
#         (z1,z2), (y1,y2), (x1,x2) = regions[i]
#         regn = np.intersect1d(antimotor, np.unique(rois[z1:z2, y1:y2, x1:x2]))
#         print('Lc* in region ->', len(regn))
#         temp[regn] = 10.0
#         weights = _draw_weights(rois, temp)
#         _draw_projections(weights, bmap, pdir / f'spmn_antimotorD_region{i}.pdf', resolution=(Rz,Ry,Rx))
#         np.save(pdir / f'regn{i}.npy', regn)
#         pca = sk.decomposition.PCA(n_components=3)
#         comps = pca.fit_transform(trcs[:,trials.ravel()][regn,:].T).T

#         plt.figure()
#         for j in range(len(comps)):
#             plt.plot(comps[j,:300] + 10*j, label=f'component {j}', alpha=0.2)
#         plt.plot(drft[:300], color='blue', label='drift')
#         plt.legend()
#         plt.savefig(pdir / f'PCA_comps{i}.pdf')
#         plt.close()

#         np.save(pdir / f'regn_comp0_{i}.npy', comps[0,:])

#         plt.figure()
#         plt.subplot(4,1,1)
#         plt.plot(drft[:Lt//4], color='blue')
#         plt.plot(comps[0,:Lt//4], color='red')
#         plt.subplot(4,1,2)
#         plt.plot(drft[Lt//4:Lt//2], color='blue')
#         plt.plot(comps[0,Lt//4:Lt//2], color='red')
#         plt.subplot(4,1,3)
#         plt.plot(drft[Lt//2:-Lt//4], color='blue')
#         plt.plot(comps[0,Lt//2:-Lt//4], color='red')
#         plt.subplot(4,1,4)
#         plt.plot(drft[-Lt//4:], color='blue')
#         plt.plot(comps[0,-Lt//4:], color='red')
#         plt.savefig(pdir / f'check{i}.pdf')
#         plt.close()

#         log.info('Analyzed dataset (%s/%s)', i+1, len(ddirs))
#         continue

        regn = np.load(pdir / f'regn{i}.npy')
        comp = np.load(pdir / f'regn_comp0_{i}.npy')
        dff = trcs[regn,:][:,trials.ravel()].mean(axis=1)
        dff2 = trcs[regn,:][:,np.where(swim > 0)[0]].mean(axis=1)
        dff3 = trcs[regn,:][:,np.where(swim <= 0)[0]].mean(axis=1)
        print(f'Lt: {Lt}, Lswim: {len(np.where(swim>0)[0])}, Lnoswim: {len(np.where(swim<=0)[0])}')

        regns.append(regn)
        comps.append(comp)
        regns_dff.append(dff)
        regns_dff2.append(dff2)
        regns_dff3.append(dff3)

        log.info('Analzing comp activity')
        corrs = np.zeros((Lc, 2), np.float32)
        for c in range(Lc):
            corrs[c,:] = sp.stats.spearmanr(trcs[c,:][trials.ravel()], comp)
        
        weights = _draw_weights(rois, corrs[:,0])
        tiff.imwrite(pdir / f'spmn_comp{i}.tif', weights, imagej=True, metadata={'axes':'ZYX'})
        _draw_projections(weights, bmap, pdir / f'spmn_comp{i}.pdf', resolution=(Rz,Ry,Rx), pmax=99.9, pmin=75)
        _draw(corrs, pdir / f'spmn_comp_hist{i}.pdf')

        antimotor = np.intersect1d(np.where(corrs[:,1] < 0.001)[0], np.where(corrs[:,0] > 0.5)[0])
        print('pcs', np.percentile(corrs[:,0], 95), np.percentile(corrs[:,0], 99), np.percentile(corrs[:,0], 99.9), np.percentile(corrs[:,0], 99.99))
        
        log.info('comp0 sig cells --> %s', len(antimotor))
        antis.append(antimotor)
        
        dff = trcs[antimotor,:][:,trials.ravel()].mean(axis=1)
        antis_dff.append(dff)

        if i == 1:
            activity = trcs[regn,:][:,trials.ravel()].mean(axis=0)
            sem = sp.stats.sem(trcs[regn,:][:,trials.ravel()], axis=0)
            mean = activity[trials[5:10].ravel()]
            sem = sem[trials[5:10].ravel()]
            plt.figure()
            plt.plot(drft[trials[5:10,:].ravel()], color='blue')
            plt.plot(mean, color='red')
            plt.fill_between(mean - sem, mean + sem, color='red', alpha=0.3)
            plt.savefig(pdir / 'average_SLOMO_activity.pdf')
            plt.close()
            break

        

        if i == 5:
            log.info('Analyzing xbrain')
            NDc  = sum([len(regns[j]) for j in range(0,3)])
            HCDc = sum([len(regns[j]) for j in range(3,6)])
            NDf  = sum([sum(regns_dff[j]) for j in range(0,3)])/NDc
            HCDf = sum([sum(regns_dff[j]) for j in range(3,6)])/HCDc
            NDf2  = sum([sum(regns_dff2[j]) for j in range(0,3)])/NDc
            HCDf2 = sum([sum(regns_dff2[j]) for j in range(3,6)])/HCDc
            NDf3  = sum([sum(regns_dff3[j]) for j in range(0,3)])/NDc
            HCDf3 = sum([sum(regns_dff3[j]) for j in range(3,6)])/HCDc

            NDdff = []
            HCDff = []
            NDdff2 = []
            HCDff2 = []
            NDdff3 = []
            HCDff3 = []
            for j in range(3):
                NDdff.extend(regns_dff[j])
                HCDff.extend(regns_dff[j+3])
                NDdff2.extend(regns_dff2[j])
                HCDff2.extend(regns_dff2[j+3])
                NDdff3.extend(regns_dff3[j])
                HCDff3.extend(regns_dff3[j+3])

            plt.figure()
            #sb.histplot(NDdff, bins=10, stat='density', kde=True, label='ND Fish', alpha=0.3, color='green', binrange=(0,1.0))
            #sb.histplot(HCDff, bins=10, stat='density', kde=True, label='HCD Fish', alpha=0.3, color='magenta', binrange=(0,1.0))
            sb.histplot(NDdff, bins=10, stat='density', kde=True, label='ND Fish', alpha=0.3, color='green')
            sb.histplot(HCDff, bins=10, stat='density', kde=True, label='HCD Fish', alpha=0.3, color='magenta')
            plt.axvline(x=NDf, color='green', linestyle='--')
            plt.axvline(x=HCDf, color='magenta', linestyle='--')
            #plt.xlim(0,1.0)
            plt.legend()
            plt.title('Distribution of Mean ΔF/F in SLOMO Neurons')
            plt.ylabel('Distribution Density')
            plt.xlabel('Mean ΔF/F')
            plt.savefig(pdir / 'SLOMO_dff.pdf')
            plt.close()

            plt.figure()
            #sb.histplot(NDdff3, bins=10, stat='density', kde=True, label='ND Fish', alpha=0.3, color='green', binrange=(0,1.0))
            #sb.histplot(HCDff3, bins=10, stat='density', kde=True, label='HCD Fish', alpha=0.3, color='magenta', binrange=(0,1.0))
            sb.histplot(NDdff3, bins=10, stat='density', kde=True, label='ND Fish', alpha=0.3, color='green')
            sb.histplot(HCDff3, bins=10, stat='density', kde=True, label='HCD Fish', alpha=0.3, color='magenta')
            plt.axvline(x=NDf3, color='green', linestyle='--')
            plt.axvline(x=HCDf3, color='magenta', linestyle='--')
            #plt.xlim(0,1.0)
            plt.legend()
            plt.title('Distribution of Mean ΔF/F in SLOMO Neurons')
            plt.ylabel('Distribution Density')
            plt.xlabel('Mean ΔF/F')
            plt.savefig(pdir / 'SLOMO_dff3.pdf')
            plt.close()

            log.info('ND  SLOMO region count: %s', NDc)
            log.info('HCD SLOMO region count: %s', HCDc)
            log.info('ND  SLOMO dF/F: %s', NDf)
            log.info('HCD SLOMO dF/F: %s', HCDf)
            log.info('ND  SLOMO dF/F2: %s', NDf2)
            log.info('HCD SLOMO dF/F2: %s', HCDf2)
            log.info('ND  SLOMO dF/F3: %s', NDf3)
            log.info('HCD SLOMO dF/F3: %s', HCDf3)

            tval, pval = sp.stats.ttest_ind(NDdff, HCDff, equal_var=False)
            log.info('t test: t: %.8f, p: %f', tval, pval)
            tval, pval = sp.stats.ttest_ind(NDdff3, HCDff3, equal_var=False)
            log.info('t test: t: %.8f, p: %f', tval, pval)
            

            continue
        continue

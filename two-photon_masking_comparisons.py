import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as st
from tqdm import tqdm
from dask import delayed
from distributed import Client, as_completed
from itertools import product

pdir = '/path/to/parent/directory'

idir = os.path.join(pdir, 'images')
Tdir = os.path.join(pdir, 'truth')
udir = os.path.join(pdir, 'untrained')
tdir = os.path.join(pdir, 'trained')
sdir = os.path.join(pdir, 'topseg')

ptch_ = sorted([f for f in os.listdir(idir) if 'tif' in f])
trth_ = sorted([f for f in os.listdir(Tdir) if 'tif' in f])
untr_ = sorted([f for f in os.listdir(udir) if 'tif' in f])
trnd_ = sorted([f for f in os.listdir(tdir) if 'tif' in f])
tpsg_ = sorted([f for f in os.listdir(sdir) if 'tif' in f])
Lf = len(ptch_)
Ly, Lx = tiff.imread(os.path.join(idir, ptch_[0])).shape

def calc_IoU(m1, m2):
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    if union == 0:
        return 0
    return inter / union

def calc_scores(trth, mask, thresh):
    tlbls = np.unique(trth)
    tlbls = tlbls[tlbls != 0]
    mlbls = np.unique(mask)
    mlbls = mlbls[mlbls != 0]

    TP = 0
    mtch = set()
    for tlbl in tlbls:
        tcmp = (trth == tlbl)
        best_IoU = 0
        best_mlbl = None 
        for mlbl in mlbls:
            if mlbl in mtch:
                continue
            mcmp = (mask == mlbl)
            _IoU = calc_IoU(tcmp, mcmp)
            if _IoU > best_IoU:
                best_IoU = _IoU
                best_mlbl = mlbl
        if best_IoU >= thresh:
            TP += 1
            mtch.add(best_mlbl)

    FP = len(mlbls) - len(mtch)
    FN = len(tlbls) - TP
    return TP, FP, FN


def calc_sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

prec_ = np.zeros((3, Lf, 100), np.float32)
recl_ = np.zeros((3, Lf, 100), np.float32)
IoU_ = np.zeros((3, Lf), np.float32)
for i in range(Lf):
    ptch = tiff.imread(os.path.join(idir, ptch_[i]))
    trth = tiff.imread(os.path.join(Tdir, trth_[i]))
    untr = tiff.imread(os.path.join(udir, untr_[i]))
    trnd = tiff.imread(os.path.join(tdir, trnd_[i]))
    tpsg = tiff.imread(os.path.join(sdir, tpsg_[i]))

    assert ptch.shape == trth.shape == untr.shape == trnd.shape
    assert all(np.unique(trth) == np.arange(0, trth.max()+1))
    assert all(np.unique(untr) == np.arange(0, untr.max()+1))
    assert all(np.unique(trnd) == np.arange(0, trnd.max()+1))
    assert all(np.unique(tpsg) == np.arange(0, tpsg.max()+1))

    IoU_[0,i] = calc_IoU(untr > 0, trth > 0)
    IoU_[1,i] = calc_IoU(trnd > 0, trth > 0)
    IoU_[2,i] = calc_IoU(tpsg > 0, trth > 0)

@delayed
def calc(f, m, t, trth, mask, thresh):
    tlbls = np.unique(trth)
    tlbls = tlbls[tlbls != 0]
    mlbls = np.unique(mask)
    mlbls = mlbls[mlbls != 0]

    TP = 0
    mtch = set()
    for tlbl in tlbls:
        tcmp = (trth == tlbl)
        best_IoU = 0
        best_mlbl = None 
        for mlbl in mlbls:
            if mlbl in mtch:
                continue
            mcmp = (mask == mlbl)
            _IoU = calc_IoU(tcmp, mcmp)
            if _IoU > best_IoU:
                best_IoU = _IoU
                best_mlbl = mlbl
        if best_IoU >= thresh:
            TP += 1
            mtch.add(best_mlbl)

    FP = len(mlbls) - len(mtch)
    FN = len(tlbls) - TP
    return f, m, t, TP, FP, FN

with Client("tcp://203.0.113.10:8080") as client:
    workers = client.scheduler_info()['workers']
    workers = workers[:np.minimum(32, len(workers))]
    print("Number of workers:", len(workers))

    threshs = np.linspace(0.01, 1.01, 100)
    Lt = len(threshs)

    masks = np.zeros((Lf,3,Ly,Lx), np.float32)
    trths = np.zeros((Lf,Ly,Lx), np.float32)
    for i in range(Lf):
        trths[i,:,:] = tiff.imread(os.path.join(Tdir, trth_[i]))  #trth
        masks[i,0,:,:] = tiff.imread(os.path.join(udir, untr_[i]))  #untr
        masks[i,1,:,:] = tiff.imread(os.path.join(tdir, trnd_[i]))  #trnd
        masks[i,2,:,:] = tiff.imread(os.path.join(sdir, tpsg_[i]))  #tpsg

    tasks = []
    for f, m, t in product(range(Lf), range(3), range(Lt)):
        tasks.append(calc(f, m, t, trths[f,:,:], masks[f,m,:,:], threshs[t]))
    jobs = client.compute(tasks, workers=workers)

    results = []
    for job in tqdm(as_completed(jobs), total=Lf*Lt*3):
        results.append({
            'f': job.result()[0],
            'm': job.result()[1],
            't': job.result()[2],
            'TP': job.result()[3],
            'FP': job.result()[4],
            'FN': job.result()[5]
        })

    prec_ = np.zeros((3, Lf, Lt), np.float32)
    recl_ = np.zeros((3, Lf, Lt), np.float32)
    f1sc_ = np.zeros((3, Lf, Lt), np.float32)
    for result in results:
        prec_[result['m'], result['f'], result['t']] = result['TP'] / (result['TP'] + result['FP'])
        recl_[result['m'], result['f'], result['t']] = result['TP'] / (result['TP'] + result['FN'])
        f1sc_[result['m'], result['f'], result['t']] = 2*result['TP'] / (2*result['TP'] + result['FN'] + result['FP'])


plt.figure()
fig, ax = plt.subplots(1,1)
mean0 = np.mean(prec_[0,:,:], axis=0)
stde0 = np.std(prec_[0,:,:], axis=0, ddof=1) / np.sqrt(prec_[0,:,:].shape[0])
ax.plot(threshs, mean0, color='green')
ax.fill_between(threshs, mean0 - stde0, mean0 + stde0, color='green', alpha=0.25)
mean1 = np.mean(prec_[1,:,:], axis=0)
stde1 = np.std(prec_[1,:,:], axis=0, ddof=1) / np.sqrt(prec_[1,:,:].shape[0])
ax.plot(threshs, mean1, color='magenta')
ax.fill_between(threshs, mean1 - stde1, mean1 + stde1, color='magenta', alpha=0.25)
mean2 = np.mean(prec_[2,:,:], axis=0)
stde2 = np.std(prec_[2,:,:], axis=0, ddof=1) / np.sqrt(prec_[2,:,:].shape[0])
ax.plot(threshs, mean2, color='orange')
ax.fill_between(threshs, mean2 - stde2, mean2 + stde2, color='orange', alpha=0.25)
ax.set_ylim(0,1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(0,1.25,0.25))
plt.savefig(os.path.join(pdir, 'precision_curve.pdf'))
plt.close()

plt.figure()
fig, ax = plt.subplots(1,1)
mean0 = np.mean(recl_[0,:,:], axis=0)
stde0 = np.std(recl_[0,:,:], axis=0, ddof=1) / np.sqrt(recl_[0,:,:].shape[0])
ax.plot(threshs, mean0, color='green')
ax.fill_between(threshs, mean0 - stde0, mean0 + stde0, color='green', alpha=0.25)
mean1 = np.mean(recl_[1,:,:], axis=0)
stde1 = np.std(recl_[1,:,:], axis=0, ddof=1) / np.sqrt(recl_[1,:,:].shape[0])
ax.plot(threshs, mean1, color='magenta')
ax.fill_between(threshs, mean1 - stde1, mean1 + stde1, color='magenta', alpha=0.25)
mean2 = np.mean(recl_[2,:,:], axis=0)
stde2 = np.std(recl_[2,:,:], axis=0, ddof=1) / np.sqrt(recl_[2,:,:].shape[0])
ax.plot(threshs, mean2, color='orange')
ax.fill_between(threshs, mean2 - stde2, mean2 + stde2, color='orange', alpha=0.25)
ax.set_ylim(0,1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(0,1.25,0.25))
plt.savefig(os.path.join(pdir, 'recall_curve.pdf'))
plt.close()

plt.figure()
fig, ax = plt.subplots(1,1)
mean0 = np.mean(f1sc_[0,:,:], axis=0)
stde0 = np.std(f1sc_[0,:,:], axis=0, ddof=1) / np.sqrt(f1sc_[0,:,:].shape[0])
ax.plot(threshs, mean0, color='green')
ax.fill_between(threshs, mean0 - stde0, mean0 + stde0, color='green', alpha=0.25)
mean1 = np.mean(f1sc_[1,:,:], axis=0)
stde1 = np.std(f1sc_[1,:,:], axis=0, ddof=1) / np.sqrt(f1sc_[1,:,:].shape[0])
ax.plot(threshs, mean1, color='magenta')
ax.fill_between(threshs, mean1 - stde1, mean1 + stde1, color='magenta', alpha=0.25)
mean2 = np.mean(f1sc_[2,:,:], axis=0)
stde2 = np.std(f1sc_[2,:,:], axis=0, ddof=1) / np.sqrt(f1sc_[2,:,:].shape[0])
ax.plot(threshs, mean2, color='orange')
ax.fill_between(threshs, mean2 - stde2, mean2 + stde2, color='orange', alpha=0.25)
ax.set_ylim(0,1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(0,1.25,0.25))
plt.savefig(os.path.join(pdir, 'F1_curve.pdf'))
plt.close()



plt.figure()
fig, ax = plt.subplots(1,1, figsize=(3,5))
p = st.ttest_rel(IoU_[0], IoU_[1], alternative='two-sided').pvalue
sig = calc_sig(p)
x1, x2 = 1, 2
y = max(np.concatenate(IoU_))
h = 0.05 * y
ax.boxplot(IoU_[0], positions=[1], vert=True, widths=0.25, medianprops=dict(color='green', linewidth=2))
ax.boxplot(IoU_[1], positions=[2], vert=True, widths=0.25, medianprops=dict(color='magenta', linewidth=2))
ax.boxplot(IoU_[2], positions=[3], vert=True, widths=0.25, medianprops=dict(color='orange', linewidth=2))

for i in range(3):
    for j in range(3):
        p = st.ttest_rel(IoU_[i], IoU_[j], alternative='two-sided').pvalue
        sig = calc_sig(p)
        print(f'{i},{j}: {sig}')

#ax.set_xlim(0,2.0)
ax.set_ylim(0,1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(pdir, 'IoU.pdf'))
plt.close()



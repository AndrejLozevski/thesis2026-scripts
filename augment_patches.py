import os
import cv2
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

pdir = '/path/to/parent/directory'

odir = os.path.join(pdir, 'train')
tdir = os.path.join(pdir, 'test')
os.makedirs(odir, exist_ok=True)
os.makedirs(tdir, exist_ok=True)

imgs = sorted([f for f in os.listdir(pdir) if 'img' in f and 'masks' not in f])
msks = sorted([f for f in os.listdir(pdir) if 'img' in f and 'masks' in f])

size = 256 #64

assert len(imgs) == len(msks) and len(imgs) > 0
for i in range(len(imgs)):
    img = tiff.imread(os.path.join(pdir, imgs[i]))
    msk = tiff.imread(os.path.join(pdir, msks[i]))
    assert img.shape == msk.shape
    Ly, Lx = img.shape

    for j in range(25):
        cY = np.random.choice(np.arange(Ly//2 - size//2, Ly//2 + size//2), 1)[0]
        cX = np.random.choice(np.arange(Lx//2 - size//2, Lx//2 + size//2), 1)[0]
        _img = img[cY - size//2:cY + size//2, cX - size//2:cX + size//2].copy()
        _msk = msk[cY - size//2:cY + size//2, cX - size//2:cX + size//2].copy()
        assert _img.shape == _msk.shape == (size, size)

        v = 0
        vals = np.unique(_msk)
        assert vals[0] == 0
        for ii in range(len(vals)):
            if np.sum(_msk == vals[ii]) < 5:
                _msk[_msk == vals[ii]] = 0
            else:
                _msk[_msk == vals[ii]] = v
                v += 1
        __ = np.unique(_msk)

        rng = np.random.choice(range(3), 1)
        if rng == 1:
            _img = np.flipud(_img)
            _msk = np.flipud(_msk)
        if rng == 2:
            _img = np.fliplr(_img)
            _msk = np.fliplr(_msk)
        assert all(np.unique(_msk) == __)
        

        rng = np.random.choice(range(4), 1)
        if rng > 0:
            _img = np.rot90(_img, k=rng)
            _msk = np.rot90(_msk, k=rng)
        assert all(np.unique(_msk) == __)

        rng = np.random.choice(range(5), 1)
        if rng < 1:
            _img = gaussian_filter(_img, sigma=2).astype(np.float32)
        elif rng < 3:
            _img = gaussian_filter(_img, sigma=1).astype(np.float32)

        rng = np.random.choice(range(3), 1)
        if rng > 1:
            _img += np.random.uniform(0, 0.125*_img.min(), size=_img.shape)
            _img = _img.astype(np.float32)

        assert _img.shape == _msk.shape == (size, size)
        assert all(np.unique(_msk) == np.arange(0, _msk.max() + 1))

        # _img = cv2.resize(_img, (_img.shape[0]*3, _img.shape[1]*3), interpolation=cv2.INTER_LINEAR)
        # _msk = np.repeat(np.repeat(_msk, 3, axis=0), 3, axis=1)
        # assert _img.shape == _msk.shape == (size*3, size*3)

        rng = np.random.choice(range(5), 1)
        _dir = odir if rng <= 3 else tdir
        tiff.imwrite(os.path.join(_dir, f'img{i:03d}_{j:02d}.tif'), _img, imagej=True, metadata={'axes':'YX'})
        tiff.imwrite(os.path.join(_dir, f'img{i:03d}_{j:02d}_masks.tif'), _msk, imagej=True, metadata={'axes':'YX'})
            


"""Feature extraction for square crops (shared by training and the app).

Pure NumPy HOG (signed gradients, soft orientation binning, 2x2-cell blocks with
L2-Hys normalisation) at two scales, plus a small intensity thumbnail.
Implemented here rather than with cv2.HOGDescriptor so results are identical
on every OpenCV version / build.
"""
import cv2
import numpy as np

SQ = 40


def _hog(imgs, cell, nbins):
    """imgs: (N, H, W) float32 -> (N, D) block-normalised signed-gradient HOG."""
    n, h, w = imgs.shape
    p = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = p[:, 1:-1, 2:] - p[:, 1:-1, :-2]
    gy = p[:, 2:, 1:-1] - p[:, :-2, 1:-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % (2 * np.pi)) * (nbins / (2 * np.pi))  # [0, nbins)
    b0 = np.floor(ang).astype(np.int32) % nbins
    b1 = (b0 + 1) % nbins
    f = ang - np.floor(ang)
    ch, cw = h // cell, w // cell
    # cell index of every pixel
    cy = np.minimum(np.arange(h) // cell, ch - 1)
    cx = np.minimum(np.arange(w) // cell, cw - 1)
    cell_id = (cy[:, None] * cw + cx[None, :]).ravel()  # (H*W,)
    size = n * ch * cw * nbins
    base = (cell_id * nbins)[None, :]
    rows = np.arange(n)[:, None] * (ch * cw * nbins)
    acc = np.zeros(size, np.float64)
    for bins, wgt in ((b0, (1 - f) * mag), (b1, f * mag)):
        idx = (rows + base + bins.reshape(n, -1)).ravel()
        acc += np.bincount(idx, weights=wgt.ravel(), minlength=size)
    hist = acc.astype(np.float32).reshape(n, ch, cw, nbins)
    blocks = []
    for y in range(ch - 1):
        for x in range(cw - 1):
            v = hist[:, y:y + 2, x:x + 2, :].reshape(n, -1)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-5)
            v = np.minimum(v, 0.2)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-5)
            blocks.append(v)
    return np.concatenate(blocks, axis=1)


def square_features(sq):
    """sq: (N, SQ, SQ) uint8 -> (N, 1686) float32."""
    sq = np.asarray(sq)
    out = []
    for i in range(0, len(sq), 2000):
        chunk = sq[i:i + 2000].astype(np.float32)
        h1 = _hog(chunk, 8, 18)                 # 5x5 cells -> 16 blocks x 72 = 1152
        h2 = _hog(chunk[:, 8:32, 8:32], 6, 12)  # centre detail: 4x4 cells -> 9 blocks x 48 = 432
        small = np.stack([cv2.resize(s, (10, 10), interpolation=cv2.INTER_AREA) for s in chunk]).reshape(len(chunk), -1) / 255.0
        m = small.mean(1, keepdims=True)
        sd = small.std(1, keepdims=True)
        out.append(np.concatenate([h1, h2, small - m, m, sd], axis=1).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 1686), np.float32)

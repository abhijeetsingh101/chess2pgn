"""Generate a dataset of square crops (grayscale, SQ x SQ) with labels 0..12."""
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

import synth

SQ = 40
sets = None


def warp_board(img, corners, sq=SQ):
    """Same preprocessing as the app: perspective-warp the board and normalise contrast."""
    size = 8 * sq
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    w = cv2.warpPerspective(g, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    lo, hi = np.percentile(w, 1), np.percentile(w, 99)
    w = np.clip((w.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return w


def squares(warped, sq=SQ):
    return warped.reshape(8, sq, 8, sq).transpose(0, 2, 1, 3).reshape(64, sq, sq)


def init(root):
    global sets
    sets = synth.load_piece_sets(root)


def job(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    img, corners, labels = synth.make_sample(sets)
    side = np.linalg.norm(corners[1] - corners[0])
    j = random.choice([0.005, 0.012, 0.02, 0.03])
    corners = corners + np.random.uniform(-j, j, (4, 2)).astype(np.float32) * side
    sq = squares(warp_board(img, corners))
    keep = [i for i in range(64) if labels[i] != 0 or random.random() < 0.3]
    return sq[keep], labels[keep]


if __name__ == "__main__":
    n, out, base = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
    with Pool(2, initializer=init, initargs=("pieces",)) as p:
        res = p.map(job, range(base, base + n), chunksize=50)
    X = np.concatenate([r[0] for r in res])
    Y = np.concatenate([r[1] for r in res])
    np.savez_compressed(out, X=X, Y=Y)
    print(out, X.shape, np.bincount(Y, minlength=13))

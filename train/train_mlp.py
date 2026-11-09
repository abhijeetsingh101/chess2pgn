"""Train a small MLP on square features in pure NumPy (Adam, ReLU, dropout).
Saves weights + feature normalisation to an .npz the app loads."""
import sys
import time

import numpy as np

sys.path.insert(0, "../app")
import features  # noqa: E402

rng = np.random.default_rng(0)


def load(paths):
    raw = [np.load(p) for p in paths]
    Y = np.concatenate([d["Y"] for d in raw])
    D = features.square_features(raw[0]["X"][:1]).shape[1]
    X = np.empty((len(Y), D), np.float16)
    o = 0
    for d in raw:
        xs = d["X"]
        for i in range(0, len(xs), 5000):
            f = features.square_features(xs[i:i + 5000])
            X[o:o + len(f)] = f
            o += len(f)
    return X, Y


def relu(x):
    return np.maximum(x, 0)


def forward(params, x, train=False, drop=0.2):
    W1, b1, W2, b2, W3, b3 = params
    h1 = relu(x @ W1 + b1)
    m1 = (rng.random(h1.shape) > drop).astype(np.float32) / (1 - drop) if train else 1
    h1d = h1 * m1
    h2 = relu(h1d @ W2 + b2)
    m2 = (rng.random(h2.shape) > drop).astype(np.float32) / (1 - drop) if train else 1
    h2d = h2 * m2
    logits = h2d @ W3 + b3
    return logits, (x, h1, m1, h1d, h2, m2, h2d)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main(train_paths, val_path, out, epochs=12, H1=512, H2=256, bs=256):
    t = time.time()
    Xtr, Ytr = load(train_paths)
    Xva, Yva = load([val_path])
    sub = Xtr[rng.choice(len(Xtr), 50000, replace=False)].astype(np.float32)
    mu, sd = sub.mean(0), sub.std(0) + 1e-4
    for i in range(0, len(Xtr), 20000):
        Xtr[i:i + 20000] = ((Xtr[i:i + 20000].astype(np.float32) - mu) / sd).astype(np.float16)
    Xva = ((Xva.astype(np.float32) - mu) / sd).astype(np.float32)
    print("features", Xtr.shape, f"{time.time() - t:.0f}s", flush=True)
    D, C = Xtr.shape[1], 13
    he = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float32)
    params = [he(D, H1), np.zeros(H1, np.float32), he(H1, H2), np.zeros(H2, np.float32),
              he(H2, C), np.zeros(C, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    n = len(Xtr)
    # class weights: empties are subsampled in data, keep roughly uniform importance
    for ep in range(epochs):
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * ep / epochs)))
        perm = rng.permutation(n)
        tl = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = Xtr[idx].astype(np.float32) + rng.standard_normal((len(idx), D)).astype(np.float32) * 0.05
            y = Ytr[idx]
            logits, (x0, h1, m1, h1d, h2, m2, h2d) = forward(params, x, True)
            p = softmax(logits)
            tl += -np.log(p[np.arange(len(y)), y] + 1e-9).sum()
            g = p
            g[np.arange(len(y)), y] -= 1
            g /= len(y)
            W1, b1, W2, b2, W3, b3 = params
            gW3 = h2d.T @ g; gb3 = g.sum(0)
            gh2 = (g @ W3.T) * m2 * (h2 > 0)
            gW2 = h1d.T @ gh2; gb2 = gh2.sum(0)
            gh1 = (gh2 @ W2.T) * m1 * (h1 > 0)
            gW1 = x0.T @ gh1; gb1 = gh1.sum(0)
            grads = [gW1 + 1e-5 * W1, gb1, gW2 + 1e-5 * W2, gb2, gW3 + 1e-5 * W3, gb3]
            step += 1
            for k in range(6):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                mh = m[k] / (1 - 0.9 ** step)
                vh = v[k] / (1 - 0.999 ** step)
                params[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        pv = forward(params, Xva)[0].argmax(1)
        acc = (pv == Yva).mean()
        print(f"epoch {ep + 1} loss {tl / n:.4f} val_acc {acc:.4f} ({time.time() - t:.0f}s)", flush=True)
    cm = np.zeros((13, 13), int)
    np.add.at(cm, (Yva, pv), 1)
    print(cm)
    np.savez_compressed(out, W1=params[0], b1=params[1], W2=params[2], b2=params[3], W3=params[4], b3=params[5],
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)

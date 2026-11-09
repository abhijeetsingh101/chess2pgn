"""
Synthetic chess-diagram generator.

Renders random positions in many visual styles -- printed-book diagrams
(hatched / grey dark squares, frames, outside coordinates), app screenshots
(flat colour themes, highlights, move dots, inside coordinates, arrows) and
photos of either (perspective, blur, lighting, noise, JPEG) -- and returns the
image together with the exact pixel corners of the 8x8 playing area.
"""
import os
import random

import cv2
import numpy as np

PIECE_NAMES = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]
# label 0 = empty, 1..12 = PIECE_NAMES order
BOOK_SETS = ["alpha", "merida", "leipzig", "cburnett", "maestro", "chess7", "companion",
             "cooke", "kosal", "mpchess", "tatiana", "staunty", "fresca", "cardinal"]

APP_THEMES = [
    ((240, 217, 181), (181, 136, 99)),   # lichess brown
    ((222, 227, 230), (140, 162, 173)),  # lichess blue
    ((238, 238, 210), (118, 150, 86)),   # chess.com green
    ((234, 233, 210), (75, 115, 153)),   # blue
    ((240, 241, 240), (196, 216, 228)),  # light blue
    ((235, 236, 208), (115, 149, 82)),
    ((255, 255, 221), (134, 166, 102)),
    ((232, 235, 239), (125, 135, 150)),  # grey
    ((241, 214, 166), (193, 141, 87)),
    ((244, 226, 197), (156, 108, 74)),
    ((220, 220, 220), (170, 170, 170)),
    ((237, 224, 255), (155, 125, 190)),  # purple
    ((250, 250, 250), (200, 200, 200)),
    ((255, 206, 158), (209, 139, 71)),
]


def load_piece_sets(root):
    sets = {}
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s)
        if not os.path.isdir(d):
            continue
        sets[s] = [cv2.imread(os.path.join(d, n + ".png"), cv2.IMREAD_UNCHANGED) for n in PIECE_NAMES]
    return sets


def put_text(img, text, org, font, scale, color, thick):
    """putText for float images (OpenCV 5 only draws on uint8)."""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(m, text, org, font, scale, 255, thick, cv2.LINE_AA)
    a = (m.astype(np.float32) / 255)[..., None]
    img[:] = img * (1 - a) + np.array(color, np.float32) * a


def rgb(c):
    return np.array(c[::-1], dtype=np.float32)  # to BGR


def random_board_labels():
    """Random (not necessarily legal) placement with realistic densities."""
    labels = np.zeros(64, dtype=np.int64)
    n = random.choice([random.randint(2, 10), random.randint(8, 24), random.randint(16, 32)])
    squares = random.sample(range(64), n)
    for sq in squares:
        labels[sq] = random.randint(1, 12)
    # pawns never on back ranks in real positions (row 0 / 7 in image coords)
    for sq in range(64):
        if labels[sq] in (1, 7) and sq // 8 in (0, 7):
            labels[sq] = random.choice([2, 3, 4, 5, 8, 9, 10, 11])
    return labels


def hatch_tile(size, light, ink, spacing, thick, angle):
    tile = np.ones((size, size, 3), np.float32) * light
    big = size * 3
    canvas = np.zeros((big, big), np.uint8)
    for x in range(-big, 2 * big, max(2, spacing)):
        cv2.line(canvas, (x, 0), (x + big, big) if angle > 0 else (x - big, big), 255, thick)
    m = canvas[size:2 * size, size:2 * size].astype(np.float32)[..., None] / 255
    return tile * (1 - m) + ink * m


def paste_piece(img, piece, x, y, s, scale=1.0, dx=0, dy=0, halo=None):
    ps = max(4, int(s * scale))
    p = cv2.resize(piece, (ps, ps), interpolation=cv2.INTER_AREA)
    a = p[:, :, 3:].astype(np.float32) / 255
    ox = x + (s - ps) // 2 + dx
    oy = y + (s - ps) // 2 + dy
    h, w = img.shape[:2]
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + ps, w), min(oy + ps, h)
    if x1 <= x0 or y1 <= y0:
        return
    pa = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    pc = p[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3].astype(np.float32)
    region = img[y0:y1, x0:x1]
    if halo is not None:  # book fonts often clear hatching around a piece
        k = max(1, ps // 14)
        hm = cv2.dilate((pa[..., 0] > 0.3).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, np.ones((ps // 4 + 1, ps // 4 + 1), np.uint8))
        region[hm > 0] = halo
    img[y0:y1, x0:x1] = pc * pa + region * (1 - pa)


def render_board(sets, labels, style):
    """Render an axis-aligned board. Returns float32 BGR image and (x0,y0,size)."""
    s = random.randint(28, 80)  # square size in px
    B = 8 * s
    if style == "book":
        pset = random.choice([b for b in BOOK_SETS if b in sets] or list(sets))
        paper = rgb(random.choice([(255, 255, 255), (250, 248, 240), (245, 240, 225), (255, 253, 245)]))
        light = paper
        ink = rgb((random.randint(0, 60),) * 3)
        mode = random.choice(["hatch", "hatch", "grey", "grey", "dots"])
        if mode == "hatch":
            sp = max(2, int(s * random.uniform(0.07, 0.16)))
            dark_tile = hatch_tile(s, light, ink, sp, random.choice([1, 1, 2]), random.choice([1, -1]))
        elif mode == "grey":
            g = random.randint(110, 215)
            dark_tile = np.ones((s, s, 3), np.float32) * rgb((g, g, g)) * (paper / 255)
        else:
            dark_tile = np.ones((s, s, 3), np.float32) * light
            st = max(2, s // random.randint(6, 12))
            for yy in range(st // 2, s, st):
                for xx in range(st // 2 + (yy // st % 2) * st // 2, s, st):
                    cv2.circle(dark_tile, (xx, yy), max(1, st // 4), ink.tolist(), -1)
        light_tile = np.ones((s, s, 3), np.float32) * light
        halo_on = mode != "grey" and random.random() < 0.5
        bg = paper
    else:
        pset = random.choice(list(sets))
        if random.random() < 0.75:
            lc, dc = random.choice(APP_THEMES)
            jit = lambda c: tuple(int(np.clip(v + random.randint(-12, 12), 0, 255)) for v in c)
            lc, dc = jit(lc), jit(dc)
        else:
            h = random.randint(0, 179)
            lc = cv2.cvtColor(np.uint8([[[h, random.randint(10, 60), random.randint(200, 250)]]]), cv2.COLOR_HSV2RGB)[0, 0]
            dc = cv2.cvtColor(np.uint8([[[h, random.randint(40, 160), random.randint(90, 180)]]]), cv2.COLOR_HSV2RGB)[0, 0]
        light_tile = np.ones((s, s, 3), np.float32) * rgb(lc)
        dark_tile = np.ones((s, s, 3), np.float32) * rgb(dc)
        if random.random() < 0.25:  # wood-ish texture
            for t in (light_tile, dark_tile):
                n = cv2.GaussianBlur(np.random.randn(s, s).astype(np.float32), (0, 0), sigmaX=s / 3, sigmaY=1)
                t += (n * random.uniform(8, 25))[..., None]
        halo_on = False
        bg = rgb(random.choice([(48, 46, 43), (22, 21, 18), (255, 255, 255), (38, 36, 33), (240, 240, 240),
                                (49, 46, 43), (60, 60, 70)]))

    margin = random.randint(int(0.15 * s), int(1.6 * s))
    W = B + 2 * margin
    img = np.ones((W, W, 3), np.float32) * bg
    x0 = y0 = margin
    dark_parity = random.choice([1, 1, 1, 0]) if style == "book" else 1
    for r in range(8):
        for c in range(8):
            t = dark_tile if (r + c) % 2 == dark_parity else light_tile
            img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s] = t

    if style != "book":
        # highlights (last move / selection / check)
        for _ in range(random.choice([0, 0, 1, 2, 2, 3])):
            sq = random.randrange(64)
            r, c = divmod(sq, 8)
            col = rgb(random.choice([(247, 247, 105), (186, 202, 68), (155, 199, 0), (255, 255, 51), (130, 151, 105)]))
            a = random.uniform(0.3, 0.7)
            reg = img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s]
            reg[:] = reg * (1 - a) + col * a
        if random.random() < 0.15:
            ks = [i for i in range(64) if labels[i] in (6, 12)]
            if ks:
                r, c = divmod(ks[0], 8)
                yy, xx = np.mgrid[0:s, 0:s]
                d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 2)
                a = np.clip(1 - d, 0, 1)[..., None] * 0.8
                reg = img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s]
                reg[:] = reg * (1 - a) + rgb((255, 0, 0)) * a

    halo = None
    for sq in range(64):
        lab = labels[sq]
        if lab == 0:
            continue
        r, c = divmod(sq, 8)
        sc = random.uniform(0.82, 1.0) if style != "book" else random.uniform(0.8, 0.98)
        jx = random.randint(-s // 20, s // 20)
        jy = random.randint(-s // 20, s // 20)
        if style == "book" and halo_on and (r + c) % 2 == dark_parity:
            halo = light
        else:
            halo = None
        paste_piece(img, sets[pset][lab - 1], x0 + c * s, y0 + r * s, s, sc, jx, jy, halo)

    if style != "book":
        # move dots on empty squares
        for _ in range(random.choice([0, 0, 0, 2, 4, 6])):
            sq = random.randrange(64)
            if labels[sq]:
                continue
            r, c = divmod(sq, 8)
            reg = img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s].copy()
            cv2.circle(reg, (s // 2, s // 2), int(s * random.uniform(0.12, 0.2)), (0, 0, 0), -1)
            img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s] = \
                img[y0 + r * s:y0 + (r + 1) * s, x0 + c * s:x0 + (c + 1) * s] * 0.8 + reg * 0.2
        # inside coordinates
        if random.random() < 0.6:
            fs = s / 90
            for i in range(8):
                col = (light_tile[0, 0] if i % 2 == 0 else dark_tile[0, 0]).tolist()
                put_text(img, "abcdefgh"[i], (x0 + i * s + int(s * 0.78), y0 + 8 * s - int(s * 0.06)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.6, col, max(1, s // 30))
                put_text(img, str(8 - i), (x0 + int(s * 0.04), y0 + i * s + int(s * 0.24)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.6, (dark_tile if i % 2 == 0 else light_tile)[0, 0].tolist(),
                            max(1, s // 30))
        # arrows
        if random.random() < 0.12:
            a1, a2 = random.sample(range(64), 2)
            p1 = (x0 + (a1 % 8) * s + s // 2, y0 + (a1 // 8) * s + s // 2)
            p2 = (x0 + (a2 % 8) * s + s // 2, y0 + (a2 // 8) * s + s // 2)
            ov = img.copy()
            cv2.arrowedLine(ov, p1, p2, rgb((21, 120, 27)).tolist(), max(2, s // 6), tipLength=0.2)
            img = img * 0.4 + ov * 0.6
    else:
        if random.random() < 0.85:  # frame
            t = random.choice([1, 1, 2, 2, 3, 4])
            gap = random.choice([0, 0, 0, 1, 2])
            cv2.rectangle(img, (x0 - gap - t // 2 - 1, y0 - gap - t // 2 - 1),
                          (x0 + B + gap + t // 2, y0 + B + gap + t // 2), ink.tolist(), t)
            if random.random() < 0.15:
                cv2.rectangle(img, (x0 - gap - t - 4, y0 - gap - t - 4),
                              (x0 + B + gap + t + 3, y0 + B + gap + t + 3), ink.tolist(), 1)
        if random.random() < 0.4 and margin > s * 0.5:  # outside coordinates
            fs = s / 110
            for i in range(8):
                put_text(img, "abcdefgh"[i], (x0 + i * s + int(s * 0.4), y0 + B + int(margin * 0.75)),
                            cv2.FONT_HERSHEY_TRIPLEX, fs * 1.4, ink.tolist(), 1)
                put_text(img, str(8 - i), (x0 - int(margin * 0.75), y0 + i * s + int(s * 0.6)),
                            cv2.FONT_HERSHEY_TRIPLEX, fs * 1.4, ink.tolist(), 1)
    return np.clip(img, 0, 255), (x0, y0, B)


def degrade(img, corners, photo):
    """Apply camera / scan / screenshot degradations. corners: 4x2 float (TL,TR,BR,BL)."""
    h, w = img.shape[:2]
    if photo:
        # perspective
        d = random.uniform(0.02, 0.12) * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.random.uniform(-d, d, (4, 2)).astype(np.float32)
        dst -= dst.min(0)
        M = cv2.getPerspectiveTransform(src, dst)
        nw, nh = int(dst[:, 0].max()) + 1, int(dst[:, 1].max()) + 1
        bgc = np.median(img[:3, :3].reshape(-1, 3), 0).tolist()
        img = cv2.warpPerspective(img, M, (nw, nh), borderValue=bgc)
        corners = cv2.perspectiveTransform(corners[None], M)[0]
        h, w = img.shape[:2]
        # lighting gradient + colour cast
        gx = np.linspace(random.uniform(0.6, 1.0), random.uniform(0.75, 1.1), w)[None, :, None]
        gy = np.linspace(random.uniform(0.7, 1.05), random.uniform(0.7, 1.05), h)[:, None, None]
        img = img * gx * gy * np.array([random.uniform(0.85, 1.05) for _ in range(3)])
        img = img * random.uniform(0.85, 1.1) + random.uniform(-15, 25)
    # resolution change
    f = random.uniform(0.35, 1.2) if photo else random.choice([1, 1, random.uniform(0.45, 1.3)])
    if f != 1:
        nw, nh = max(64, int(w * f)), max(64, int(h * f))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        corners = corners * np.array([nw / w, nh / h])
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) if random.random() < 0.2 else img
        if img.shape[1] == w:
            corners = corners * np.array([w / nw, h / nh])
    if random.random() < (0.7 if photo else 0.3):
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.4 if photo else 0.8))
    if random.random() < (0.8 if photo else 0.2):
        img = img + np.random.randn(*img.shape) * random.uniform(1, 9)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.6:
        q = random.randint(25, 95)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    if random.random() < 0.15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img, corners.astype(np.float32)


def make_sample(sets, style=None, photo=None):
    style = style or random.choice(["book", "book", "app", "app", "app"])
    photo = (random.random() < 0.3) if photo is None else photo
    labels = random_board_labels()
    img, (x0, y0, B) = render_board(sets, labels, style)
    corners = np.float32([[x0, y0], [x0 + B, y0], [x0 + B, y0 + B], [x0, y0 + B]])
    img, corners = degrade(img, corners, photo)
    return img, corners, labels

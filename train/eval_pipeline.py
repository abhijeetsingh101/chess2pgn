"""End-to-end evaluation on synthetic pages with known ground truth."""
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "../app")
import recognize as R  # noqa: E402
import synth  # noqa: E402

sets = synth.load_piece_sets("pieces")


def make_page(kind):
    """Returns image, list of (corners, labels)."""
    if kind == "bookpage":
        W, H = 1240, 1754
        page = np.full((H, W, 3), 255, np.uint8)
        for y in range(60, H - 40, 34):  # fake text lines
            if random.random() < 0.7:
                cv2.rectangle(page, (80, y), (80 + random.randint(300, 1080), y + 10), (90, 90, 90), -1)
        gts = []
        n = random.choice([1, 2, 4, 6])
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        cell_w, cell_h = (W - 160) // cols, (H - 200) // rows
        for i in range(n):
            labels = synth.random_board_labels()
            bimg, (x0, y0, B) = synth.render_board(sets, labels, "book")
            sc = min(cell_w * 0.85 / bimg.shape[1], cell_h * 0.8 / bimg.shape[0])
            bimg = cv2.resize(bimg.astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            ox = 80 + c * cell_w + (cell_w - bimg.shape[1]) // 2
            oy = 100 + r * cell_h + 10
            page[oy:oy + bimg.shape[0], ox:ox + bimg.shape[1]] = bimg
            # blank out text under the board region margins already covered by board image background
            corners = np.float32([[x0, y0], [x0 + B, y0], [x0 + B, y0 + B], [x0, y0 + B]]) * sc + [ox, oy]
            gts.append((corners, labels))
        photo = random.random() < 0.3
        if photo:
            allc = np.concatenate([g[0] for g in gts])
            img, allc = synth.degrade(page.astype(np.float32), allc, True)
            gts = [(allc[4 * i:4 * i + 4], gts[i][1]) for i in range(len(gts))]
            return img, gts
        return page, gts
    else:  # app screenshot / photo of screen
        labels = synth.random_board_labels()
        bimg, (x0, y0, B) = synth.render_board(sets, labels, "app")
        bimg = bimg.astype(np.uint8)
        W = int(bimg.shape[1] * random.uniform(1.0, 2.2))
        H = int(bimg.shape[0] * random.uniform(1.0, 1.8))
        bg = bimg[1, 1].tolist()
        page = np.full((H, W, 3), bg, np.uint8)
        # UI clutter
        for _ in range(random.randint(0, 6)):
            x, y = random.randint(0, W - 20), random.randint(0, H - 20)
            cv2.rectangle(page, (x, y), (x + random.randint(20, 300), y + random.randint(10, 60)),
                          [random.randint(0, 255)] * 3, -1)
        ox, oy = random.randint(0, W - bimg.shape[1]), random.randint(0, H - bimg.shape[0])
        page[oy:oy + bimg.shape[0], ox:ox + bimg.shape[1]] = bimg
        corners = np.float32([[x0, y0], [x0 + B, y0], [x0 + B, y0 + B], [x0, y0 + B]]) + [ox, oy]
        img, corners = synth.degrade(page.astype(np.float32), corners, kind == "appphoto")
        return img, [(corners, labels)]


def main(n=40, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    stats = {}
    for kind in ["bookpage", "app", "appphoto"]:
        found = total = fp = 0
        sq_ok = sq_tot = boards_perfect = 0
        t = time.time()
        for _ in range(n):
            img, gts = make_page(kind)
            dets = R.detect_boards(img)
            used = set()
            for corners, labels in gts:
                side = np.sqrt(R.quad_area(corners))
                best, bi = 1e9, -1
                for i, d in enumerate(dets):
                    e = np.abs(R.order_corners(d["corners"]) - corners).max() / side
                    if e < best:
                        best, bi = e, i
                total += 1
                if best < 0.06:
                    found += 1
                    used.add(bi)
                    lab, conf, _ = R.recognize(img, dets[bi]["corners"])
                    ok = (lab == labels).sum()
                    sq_ok += ok
                    sq_tot += 64
                    boards_perfect += ok == 64
            fp += len(dets) - len(used)
        stats[kind] = dict(found=f"{found}/{total}", false_pos=fp, sq_acc=round(sq_ok / max(sq_tot, 1), 4),
                           perfect_boards=f"{boards_perfect}/{found}", sec_per_img=round((time.time() - t) / n, 2))
        print(kind, stats[kind], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

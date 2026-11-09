"""
Board detection + piece recognition.

detect_boards(img)       -> list of candidate boards (4 corner points + score)
recognize(img, corners)  -> 8x8 piece labels + confidences
process_image / process_pdf -> list of puzzle dicts ready for the web UI
"""
import os
import re

import chess
import cv2
import numpy as np

from features import SQ, square_features

LABELS = [None, "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
_model = None


# ----------------------------------------------------------------------------- model
def load_model(path=None):
    global _model
    if _model is None:
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")
        _model = dict(np.load(path))
    return _model


def predict_squares(sqs):
    m = load_model()
    x = (square_features(sqs) - m["mu"]) / m["sd"]
    h = np.maximum(x @ m["W1"] + m["b1"], 0)
    h = np.maximum(h @ m["W2"] + m["b2"], 0)
    z = h @ m["W3"] + m["b3"]
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- geometry
def order_corners(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise from ~(-180deg) in image coords: TL first-ish
    i = np.argmin(pts.sum(1))
    return np.roll(pts, -i, axis=0)  # TL, TR, BR, BL


def warp_gray(gray, corners, size):
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(gray, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def warp_board(img, corners, sq=SQ):
    """Identical to the training-time preprocessing."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    w = warp_gray(g, corners, 8 * sq)
    lo, hi = np.percentile(w, 1), np.percentile(w, 99)
    return np.clip((w.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def quad_area(q):
    return abs(cv2.contourArea(np.float32(q)))


# ----------------------------------------------------------------------------- grid score
_N = 256
_P = _N // 8


def grid_score(gray, corners):
    """How well does this quad line up with an 8x8 grid?  ~0 = not at all, larger = better."""
    w = warp_gray(gray, corners, _N).astype(np.float32)
    b = cv2.GaussianBlur(w, (0, 0), 2.2)
    lo, hi = np.percentile(b, 3), np.percentile(b, 97)
    c = hi - lo
    if c < 20:
        return 0.0
    gx = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)) / c
    gy = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / c
    profs = (np.median(gx, axis=0), np.median(gy, axis=1), gx.mean(0), gy.mean(1))
    sc = []
    for prof in profs:
        peaks = np.array([prof[k * _P - 2:k * _P + 3].max() for k in range(1, 8)])
        mids = np.array([prof[k * _P + _P // 2 - 5:k * _P + _P // 2 + 6].mean() for k in range(0, 8)])
        strength = 0.5 * np.sort(peaks)[1:].mean() + 0.5 * peaks.mean()
        sc.append(strength / (np.median(mids) + 0.04) * min(1.0, strength / 0.08))
    m = cv2.resize(w, (8, 8), interpolation=cv2.INTER_AREA)
    chk = np.indices((8, 8)).sum(0) % 2 * 2 - 1
    d = m - np.median(m)
    par = abs((d * chk).sum()) / (np.abs(d).sum() + 1e-3)
    return float((min(sc[0], sc[1]) * 0.35 + min(sc[2], sc[3]) * 0.35
                  + (sc[0] + sc[1] + sc[2] + sc[3]) * 0.075) * (0.6 + 0.8 * par))


def refine(gray, corners, iters=3):
    q = order_corners(corners).copy()
    side = np.sqrt(quad_area(q))
    best = grid_score(gray, q)
    for step_frac in [0.04, 0.02, 0.01, 0.005][: iters + 1]:
        step = max(0.5, side * step_frac)
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            # whole-quad moves: translate / scale
            moves = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                moves.append(q + np.float32([dx, dy]) * step)
            c = q.mean(0)
            for s in (-1, 1):
                moves.append(c + (q - c) * (1 + s * step / side * 2))
            # single-corner moves
            for i in range(4):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    qq = q.copy()
                    qq[i] += np.float32([dx, dy]) * step
                    moves.append(qq)
            for qq in moves:
                s_ = grid_score(gray, qq)
                if s_ > best + 1e-4:
                    best, q, improved = s_, qq.astype(np.float32), True
    return q, best


def fit_lattice(gray, corners, N=480):
    """Inside a (possibly too large) quad, find the 8x8 grid by fitting 9 evenly spaced lines per axis."""
    w = warp_gray(gray, corners, N).astype(np.float32)
    b = cv2.GaussianBlur(w, (0, 0), 2.0)
    c = np.percentile(b, 97) - np.percentile(b, 3)
    if c < 20:
        return corners
    gx = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)) / c
    gy = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / c
    res = []
    for prof in (np.median(gx, 0) + gx.mean(0), np.median(gy, 1) + gy.mean(1)):
        prof = np.maximum.reduce([np.roll(prof, k) for k in (-1, 0, 1)])  # tolerate 1px misalignment
        best, bo, bp = -1, 0, N / 8
        for p in np.arange(N * 0.5 / 8, N / 8 + 0.01, 0.5):
            offs = np.arange(0, N - 8 * p + 0.01, 1.0)
            if len(offs) == 0:
                continue
            ks = np.arange(0, 9)
            idx = np.clip(np.round(offs[:, None] + ks[None] * p).astype(int), 0, N - 1)
            mid = np.clip(np.round(offs[:, None] + (np.arange(8)[None] + 0.5) * p).astype(int), 0, N - 1)
            pk = prof[idx]
            vals = 0.6 * pk[:, 1:8].mean(1) + 0.4 * pk[:, [0, 8]].mean(1) - np.median(prof[mid], 1)
            i = int(np.argmax(vals))
            if vals[i] > best:
                best, bo, bp = vals[i], offs[i], p
        res.append((bo, bo + 8 * bp))
    (x0, x1), (y0, y1) = res
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [N, 0], [N, N], [0, N]]), np.float32(corners))
    return cv2.perspectiveTransform(src[None], M)[0]


# ----------------------------------------------------------------------------- detection
def _candidates(gray):
    h, w = gray.shape
    min_side = 0.08 * min(h, w)
    min_side = max(min_side, 60)
    cands = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    maps = []
    ed = cv2.Canny(blur, 30, 90)
    maps.append(cv2.dilate(ed, np.ones((3, 3), np.uint8)))
    maps.append(cv2.dilate(cv2.Canny(blur, 12, 36), np.ones((3, 3), np.uint8)))
    maps.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8))
    _, ot = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps += [ot, 255 - ot]
    # board-vs-background: strong closing merges the squares of the board into one blob
    k = max(3, int(min(h, w) * 0.01)) | 1
    maps.append(cv2.morphologyEx(maps[0], cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)))
    # frameless diagrams: fill hatching so dark squares (touching at corners) form one blob
    paper = np.percentile(blur, 90)
    for ink in (255 - ot, ((blur < paper - 14) * 255).astype(np.uint8)):
        for kk in (3, 7, 13):
            maps.append(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8)))
    for mi, mp in enumerate(maps):
        cnts, _ = cv2.findContours(mp, cv2.RETR_LIST if mi < 6 else cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_side ** 2:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.04):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    break
            else:
                ap = cv2.boxPoints(cv2.minAreaRect(hull))
            if mi >= 6:
                box = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
                bs = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
                if min(bs) >= min_side and max(bs) / min(bs) < 1.8:
                    cands.append(box)
            q = order_corners(ap)
            sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
            if min(sides) < min_side or max(sides) / min(sides) > 1.8:
                continue
            if quad_area(q) / max(cv2.contourArea(hull), 1) < 0.85:
                continue
            cands.append(q)
    cands.append(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))  # whole image (cropped screenshots)
    return cands


def _iou_like(a, b):
    """Overlap of two quads relative to the smaller one."""
    ra, rb = cv2.boundingRect(np.int32(a)), cv2.boundingRect(np.int32(b))
    x0, y0 = max(ra[0], rb[0]), max(ra[1], rb[1])
    x1, y1 = min(ra[0] + ra[2], rb[0] + rb[2]), min(ra[1] + ra[3], rb[1] + rb[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1, min(ra[2] * ra[3], rb[2] * rb[3]))


def _reading_order(items):
    """Group boards into rows (vertical overlap), then left-to-right within a row."""
    items = sorted(items, key=lambda t: t[1][:, 1].min())
    rows = []
    for it in items:
        y0, y1 = it[1][:, 1].min(), it[1][:, 1].max()
        for row in rows:
            ry0, ry1 = row["y0"], row["y1"]
            if min(y1, ry1) - max(y0, ry0) > 0.5 * min(y1 - y0, ry1 - ry0):
                row["items"].append(it)
                row["y0"], row["y1"] = min(ry0, y0), max(ry1, y1)
                break
        else:
            rows.append({"y0": y0, "y1": y1, "items": [it]})
    return [it for row in rows for it in sorted(row["items"], key=lambda t: t[1][:, 0].min())]


def detect_boards(img, max_dim=1600, min_score=3.0):
    """Return list of {"corners": 4x2 array in original pixel coords, "score": float}."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = min(1.0, max_dim / max(gray.shape))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    cands = _candidates(g)
    # quick score, dedupe, then refine the promising ones
    scored = []
    for q in cands:
        sc = grid_score(g, q)
        lq = fit_lattice(g, q)
        if np.sqrt(quad_area(lq)) > 40:
            ls = grid_score(g, lq)
            if ls > sc:
                sc, q = ls, lq
        scored.append((sc, q))
    scored.sort(key=lambda t: -t[0])
    kept = []
    for sc, q in scored:
        if sc < min_score * 0.6:
            break
        if any(_iou_like(q, k) > 0.9 and abs(quad_area(q) / quad_area(k) - 1) < 0.15 for _, k in kept):
            continue
        kept.append((sc, q))
        if len(kept) > 70:
            break
    refined = []
    for sc, q in kept:
        rq, rs = refine(g, q)
        if rs >= min_score:
            refined.append((rs, rq))
    refined.sort(key=lambda t: -t[0])
    out = []
    for rs, rq in refined:
        if any(_iou_like(rq, o) > 0.35 for _, o in out):
            continue
        out.append((rs, rq))
    # reading order: top-to-bottom, left-to-right (columns on book pages)
    out = _reading_order(out)
    return [{"corners": (q / s).astype(np.float32), "score": float(sc)} for sc, q in out]


# ----------------------------------------------------------------------------- recognition
def recognize(img, corners):
    w = warp_board(img, corners)
    sqs = w.reshape(8, SQ, 8, SQ).transpose(0, 2, 1, 3).reshape(64, SQ, SQ)
    p = predict_squares(sqs)
    lab = p.argmax(1)
    return lab, p.max(1), p


def guess_flipped(lab):
    """True if the board appears to be shown from Black's side."""
    rows = np.arange(64) // 8
    wp, bp = rows[lab == 1], rows[lab == 7]
    wk, bk = rows[lab == 6], rows[lab == 12]
    ev, n = 0.0, 0
    if len(wp) and len(bp):
        ev += wp.mean() - bp.mean(); n += 1
    elif len(wp):
        ev += (wp.mean() - 3.5) * 0.6; n += 1
    elif len(bp):
        ev += (3.5 - bp.mean()) * 0.6; n += 1
    if len(wk) and len(bk):
        ev += 0.5 * (wk[0] - bk[0]) / 7 * 3; n += 1
    return n > 0 and ev < -1.5


def labels_to_placement(lab, flipped=False):
    grid = np.array([LABELS[x] for x in lab], dtype=object).reshape(8, 8)
    if flipped:
        grid = grid[::-1, ::-1]
    rows = []
    for r in range(8):
        s, e = "", 0
        for c in range(8):
            if grid[r, c] is None:
                e += 1
            else:
                if e:
                    s += str(e); e = 0
                s += grid[r, c]
        if e:
            s += str(e)
        rows.append(s)
    return "/".join(rows)


def guess_castling(placement):
    b = chess.Board(placement + " w - - 0 1")
    rights = ""
    if b.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if b.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE): rights += "K"
        if b.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE): rights += "Q"
    if b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if b.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK): rights += "k"
        if b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK): rights += "q"
    return rights or "-"


def check_position(fen):
    """Return list of human-readable problems ([] if the position is legal)."""
    try:
        b = chess.Board(fen)
    except ValueError as e:
        return [str(e)]
    st = b.status()
    msgs = {
        chess.STATUS_NO_WHITE_KING: "No white king", chess.STATUS_NO_BLACK_KING: "No black king",
        chess.STATUS_TOO_MANY_KINGS: "Too many kings", chess.STATUS_TOO_MANY_WHITE_PAWNS: "Too many white pawns",
        chess.STATUS_TOO_MANY_BLACK_PAWNS: "Too many black pawns", chess.STATUS_PAWNS_ON_BACKRANK: "Pawn on 1st/8th rank",
        chess.STATUS_TOO_MANY_WHITE_PIECES: "Too many white pieces", chess.STATUS_TOO_MANY_BLACK_PIECES: "Too many black pieces",
        chess.STATUS_BAD_CASTLING_RIGHTS: "Castling rights don't match the position",
        chess.STATUS_INVALID_EP_SQUARE: "Invalid en passant square",
        chess.STATUS_OPPOSITE_CHECK: "Side not to move is in check (wrong side to move?)",
        chess.STATUS_TOO_MANY_CHECKERS: "Too many checking pieces", chess.STATUS_IMPOSSIBLE_CHECK: "Impossible check",
    }
    return [t for f, t in msgs.items() if st & f]


# ----------------------------------------------------------------------------- captions (PDF text)
_STM = re.compile(r"\b(white|black)\s+(to\s+(move|play)|plays|moves|wins|draws)\b", re.I)
_STM_SHORT = re.compile(r"(?<![a-z])\(?\s*(w|b)\s*\)?\s*$", re.I)


def side_from_text(text):
    m = _STM.search(text or "")
    if m:
        return "w" if m.group(1).lower() == "white" else "b"
    if re.search(r"[☐□○◯]\s*$", text or ""):
        return "w"
    if re.search(r"[■●◼]\s*$", text or ""):
        return "b"
    return None


# ----------------------------------------------------------------------------- top level
def board_to_puzzle(img, det, caption="", source=""):
    lab, conf, _ = recognize(img, det["corners"])
    flipped = guess_flipped(lab)
    placement = labels_to_placement(lab, flipped)
    # puzzle apps/books show the board from the solver's side, so a flipped board usually means Black to move
    stm = side_from_text(caption) or ("b" if flipped else "w")
    fen = f"{placement} {stm} {guess_castling(placement)} - 0 1"
    low = [int(i) for i in np.where(conf < 0.8)[0]]
    if flipped:
        low = [63 - i for i in low]
    q = det["corners"]
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    pad = int(0.04 * max(x1 - x0, y1 - y0))
    H, W = img.shape[:2]
    crop = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
    return {
        "fen": fen,
        "flipped": bool(flipped),
        "uncertain": low,  # square indices (a8=0 .. h1=63) the model was unsure about
        "confidence": float(conf.min()),
        "caption": caption.strip()[:300],
        "source": source,
        "score": det["score"],
        "corners": q.tolist(),
        "crop": crop,
        "problems": check_position(fen),
    }


def process_image(img, source="", manual_corners=None):
    if manual_corners is not None:
        q = order_corners(manual_corners)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rq, rs = refine(gray, q, iters=2)
        # only accept the refinement if it didn't wander off
        if np.abs(rq - q).max() < 0.08 * np.sqrt(quad_area(q)):
            q = rq
        dets = [{"corners": q, "score": rs}]
    else:
        dets = detect_boards(img)
    return [board_to_puzzle(img, d, source=source) for d in dets]


def process_pdf_page(doc, pno, dpi=170):
    """Render one page of an open PyMuPDF document; returns (puzzles, page_image)."""
    page = doc[pno]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    dets = detect_boards(img)
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wno (PDF points)
    z = dpi / 72.0
    puzzles = []
    for d in dets:
        q = d["corners"] / z
        bx0, by0 = q.min(0)
        bx1, by1 = q.max(0)
        hgt = by1 - by0
        # caption text: just below and just above the diagram, roughly the same horizontal span
        span = [w for w in words if w[0] < bx1 + 10 and w[2] > bx0 - 10]
        below = sorted([w for w in span if by1 - 2 < w[1] < by1 + 0.3 * hgt], key=lambda w: (w[1], w[0]))
        above = sorted([w for w in span if by0 - 0.25 * hgt < w[3] < by0 + 2], key=lambda w: (w[1], w[0]))
        cap_below = " ".join(w[4] for w in below)
        cap_above = " ".join(w[4] for w in above)
        cap = cap_below if (side_from_text(cap_below) or not side_from_text(cap_above)) and cap_below else cap_above
        puzzles.append(board_to_puzzle(img, d, caption=cap or cap_below, source=f"page {pno + 1}"))
    return puzzles, img


def process_pdf(path_or_bytes, dpi=170, max_pages=300):
    import pymupdf as fitz  # PyMuPDF

    doc = fitz.open(stream=path_or_bytes, filetype="pdf") if isinstance(path_or_bytes, (bytes, bytearray)) \
        else fitz.open(path_or_bytes)
    puzzles = []
    for pno in range(min(len(doc), max_pages)):
        puzzles += process_pdf_page(doc, pno, dpi)[0]
    return puzzles


def make_pgn(puzzles, event="Puzzles"):
    """puzzles: list of {fen, solution, title, result?} -> PGN text (one game per puzzle)."""
    import chess.pgn
    import io

    out = io.StringIO()
    for i, p in enumerate(puzzles, 1):
        board = chess.Board(p["fen"])
        g = chess.pgn.Game()
        g.headers["Event"] = p.get("title") or f"{event} #{i}"
        g.headers["Site"] = p.get("source") or "?"
        g.headers["Date"] = "????.??.??"
        g.headers["Round"] = str(i)
        g.headers["White"] = "?"
        g.headers["Black"] = "?"
        g.setup(board)
        node = g
        b = board.copy()
        sol = (p.get("solution") or "").strip()
        bad = None
        if sol:
            toks = re.sub(r"\{[^}]*\}", " ", sol)
            toks = re.sub(r"\d+\.(\.\.)?", " ", toks).replace("…", " ").split()
            for t in toks:
                if t in ("1-0", "0-1", "1/2-1/2", "*") or re.fullmatch(r"[!?]+", t):
                    continue
                try:
                    mv = b.parse_san(t.rstrip("!?"))
                except ValueError:
                    bad = t
                    break
                node = node.add_variation(mv)
                b.push(mv)
        res = p.get("result") or ("1-0" if b.is_checkmate() and b.turn == chess.BLACK else
                                  "0-1" if b.is_checkmate() and b.turn == chess.WHITE else "*")
        g.headers["Result"] = res
        comment = (p.get("comment") or "").strip()
        if bad:
            comment = (comment + f" [unparsed solution from '{bad}': {sol}]").strip()
        if comment:
            g.comment = comment
        print(g, file=out, end="\n\n")
    return out.getvalue()

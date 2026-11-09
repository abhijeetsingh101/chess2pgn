"""
Chess Diagram -> PGN  (Streamlit version)

Local:   streamlit run streamlit_app.py
Cloud:   push this folder to GitHub, then deploy it on https://share.streamlit.io
"""
import base64
import re
import sys
import uuid
from pathlib import Path

import chess
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "app"))
import recognize as R  # noqa: E402

st.set_page_config(page_title="Chess Diagram → PGN", page_icon="♞", layout="wide")

_board = components.declare_component("board_editor", path=str(ROOT / "components" / "board"))
_select = components.declare_component("region_select", path=str(ROOT / "components" / "select"))

PER_PAGE = 10
MAX_PDF_PAGES = 400
ss = st.session_state


@st.cache_resource
def _load_model():
    return R.load_model()


_load_model()

# ----------------------------------------------------------------------------- state
for k, v in {"puzzles": [], "files": {}, "pages": [], "counter": 0, "sel_seq": None, "list_page": 1, "upk": 0}.items():
    ss.setdefault(k, v)


def png_bytes(img, max_side=700):
    if max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


def jpg_bytes(img, max_side=2000):
    if max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), img.shape[1], img.shape[0]


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return img


def add_page(img, label):
    jpg, w, h = jpg_bytes(img)
    pid = uuid.uuid4().hex[:8]
    ss.pages.append({"id": pid, "label": label, "jpg": jpg, "w": w, "h": h})
    return pid


def add_puzzle(raw):
    ss.counter += 1
    pid = uuid.uuid4().hex[:10]
    parts = raw["fen"].split()
    m = re.match(r"^\s*(?:No\.?|#|Diagram|Puzzle|Exercise|Problem)?\s*(\d{1,4})\b", raw.get("caption") or "", re.I)
    title = f"Puzzle {m.group(1)}" if m else f"Puzzle {ss.counter}"
    p = {
        "id": pid, "placement": parts[0], "uncertain": list(raw.get("uncertain", [])), "flipped": raw.get("flipped", False),
        "caption": raw.get("caption", ""), "source": raw.get("source", ""), "crop": png_bytes(raw["crop"]),
        "version": 0, "seq": None, "fen_shown": None, "fen_error": None,
    }
    ss[f"title_{pid}"] = title
    ss[f"stm_{pid}"] = "Black" if parts[1] == "b" else "White"
    for k in "KQkq":
        ss[f"cs{k}_{pid}"] = k in parts[2]
    ss[f"sol_{pid}"] = ""
    ss[f"inc_{pid}"] = True
    ss.puzzles.append(p)


def castling_possible(placement):
    b = chess.Board(placement + " w - - 0 1")
    K, R_ = chess.Piece(chess.KING, chess.WHITE), chess.Piece(chess.ROOK, chess.WHITE)
    k, r = chess.Piece(chess.KING, chess.BLACK), chess.Piece(chess.ROOK, chess.BLACK)
    return {
        "K": b.piece_at(chess.E1) == K and b.piece_at(chess.H1) == R_,
        "Q": b.piece_at(chess.E1) == K and b.piece_at(chess.A1) == R_,
        "k": b.piece_at(chess.E8) == k and b.piece_at(chess.H8) == r,
        "q": b.piece_at(chess.E8) == k and b.piece_at(chess.A8) == r,
    }


def current_fen(p):
    pid = p["id"]
    poss = castling_possible(p["placement"])
    cs = "".join(k for k in "KQkq" if poss[k] and ss.get(f"cs{k}_{pid}")) or "-"
    stm = "b" if ss.get(f"stm_{pid}") == "Black" else "w"
    return f"{p['placement']} {stm} {cs} - 0 1"


def check_solution(fen, sol):
    """Returns (error or None, SAN string)."""
    sol = (sol or "").strip()
    if not sol:
        return None, ""
    import io
    import chess.pgn
    pgn = R.make_pgn([{"fen": fen, "solution": sol}])
    err = None
    if "[unparsed solution from '" in pgn:
        tok = pgn.split("[unparsed solution from '")[1].split("'")[0]
        err = f"“{tok}” is not a legal move here"
    g = chess.pgn.read_game(io.StringIO(pgn))
    san = chess.Board(fen).variation_san(list(g.mainline_moves())) if g and list(g.mainline_moves()) else ""
    return err, san


def needs_attention(p):
    fen = current_fen(p)
    if R.check_position(fen):
        return True
    if p["uncertain"]:
        return True
    return check_solution(fen, ss.get(f"sol_{p['id']}"))[0] is not None


def remove_puzzle(pid):
    ss.puzzles = [p for p in ss.puzzles if p["id"] != pid]


# ----------------------------------------------------------------------------- header
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      div[data-testid="stImage"] img { border: 1px solid #e2ddd5; border-radius: 6px; }
      .small-muted { color: #77726a; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("♞ Chess Diagram → PGN")
st.caption("Book pages, PDFs, screenshots and photos in — clean PGN out. Each diagram is found automatically; "
           "check each board, then download the PGN from the sidebar.")

# ----------------------------------------------------------------------------- upload + processing
uploads = st.file_uploader("Upload PDFs or images", type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                           accept_multiple_files=True, key=f"uploader_{ss.upk}",
                           help="PDF books, scans, phone photos or app screenshots")

for f in uploads or []:
    fid = getattr(f, "file_id", None) or f"{f.name}-{f.size}"
    if fid not in ss.files:
        ss.files[fid] = {"name": f.name, "next": 0, "total": None, "done": False}

pending = [(f, ss.files[getattr(f, "file_id", None) or f"{f.name}-{f.size}"]) for f in (uploads or [])]
pending = [(f, info) for f, info in pending if not info["done"]]
if pending:
    prog = st.progress(0.0, text="Starting…")
    for f, info in pending:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            import pymupdf

            doc = pymupdf.open(stream=data, filetype="pdf")
            info["total"] = min(len(doc), MAX_PDF_PAGES)
            while info["next"] < info["total"]:   # resumable: progress is saved page by page
                pno = info["next"]
                prog.progress(pno / info["total"], text=f"{f.name}: page {pno + 1} of {info['total']} · "
                                                         f"{len(ss.puzzles)} diagrams so far")
                puz, page_img = R.process_pdf_page(doc, pno)
                add_page(page_img, f"{f.name} – page {pno + 1}")
                for p in puz:
                    p["source"] = f"{f.name}, page {pno + 1}"
                    add_puzzle(p)
                info["next"] = pno + 1
        else:
            prog.progress(0.5, text=f"{f.name}…")
            img = decode_image(data)
            if img is None:
                st.warning(f"Could not read {f.name}")
            else:
                add_page(img, f.name)
                puz = R.process_image(img, source=f.name)
                if not puz:
                    st.warning(f"No board found in {f.name}. Use “Missed a diagram?” below to select it by hand.")
                for p in puz:
                    add_puzzle(p)
        info["done"] = True
    prog.empty()
    st.toast(f"{len(ss.puzzles)} diagram(s) ready")

# ----------------------------------------------------------------------------- manual selection
if ss.pages:
    with st.expander("Missed a diagram? Select it by hand"):
        labels = [pg["label"] for pg in ss.pages]
        i = st.selectbox("Page", range(len(labels)), format_func=lambda j: labels[j], key="manual_page")
        pg = ss.pages[i]
        url = "data:image/jpeg;base64," + base64.b64encode(pg["jpg"]).decode()
        val = _select(image=url, key=f"select_{pg['id']}", default=None)
        if val and val.get("box") and val.get("seq") != ss.sel_seq:
            ss.sel_seq = val["seq"]
            x0, y0, x1, y1 = val["box"]
            img = cv2.imdecode(np.frombuffer(pg["jpg"], np.uint8), cv2.IMREAD_COLOR)
            W, H = pg["w"], pg["h"]
            corners = np.float32([[x0 * W, y0 * H], [x1 * W, y0 * H], [x1 * W, y1 * H], [x0 * W, y1 * H]])
            for p in R.process_image(img, source=pg["label"], manual_corners=corners):
                add_puzzle(p)
            st.toast("Board added at the end of the list")

# ----------------------------------------------------------------------------- cards
if not ss.puzzles:
    st.info(
        "**How it works**\n\n"
        "1. **Upload** a PDF of a puzzle book, scanned pages, phone photos, or app screenshots.\n"
        "2. **Review** each recognised board next to the original — fix any piece, set who is to move.\n"
        "3. Optionally type the **solution** (e.g. `1. Qxf7+ Kxf7 2. Bc4+`); it is checked for legality.\n"
        "4. **Download** one PGN with every puzzle from the sidebar — ready for Lichess Studies, ChessBase, SCID…"
    )
else:
    top = st.columns([2, 2, 1])
    only_bad = top[0].toggle("Show only boards needing attention", key="only_bad")
    shown = [p for p in ss.puzzles if needs_attention(p)] if only_bad else list(ss.puzzles)
    n_pages = max(1, (len(shown) + PER_PAGE - 1) // PER_PAGE)
    ss.list_page = min(ss.list_page, n_pages)
    if n_pages > 1:
        ss.list_page = top[1].number_input(f"Page (of {n_pages})", 1, n_pages, ss.list_page)
    top[2].markdown(f"<div class='small-muted' style='text-align:right;padding-top:.5rem'>"
                    f"{len(ss.puzzles)} boards</div>", unsafe_allow_html=True)

    start = (ss.list_page - 1) * PER_PAGE
    for p in shown[start:start + PER_PAGE]:
        pid = p["id"]
        num = ss.puzzles.index(p) + 1
        fkey = f"fen_{pid}"

        # latest edit made in the board editor (stored by Streamlit under the component key)
        val = ss.get(f"board_{pid}_{p['version']}")
        if val and val.get("seq") != p["seq"] and val.get("version") == p["version"]:
            p["seq"] = val["seq"]
            p["placement"] = val["placement"]
            p["uncertain"] = val.get("uncertain", [])
            p["flipped"] = val.get("flipped", p["flipped"])

        # a FEN typed by hand replaces the board
        if fkey in ss and p["fen_shown"] is not None and ss[fkey] != p["fen_shown"]:
            try:
                b = chess.Board(ss[fkey].strip())
                p["placement"] = b.board_fen()
                ss[f"stm_{pid}"] = "White" if b.turn else "Black"
                cr = ss[fkey].split()[2] if len(ss[fkey].split()) > 2 else "-"
                for k in "KQkq":
                    ss[f"cs{k}_{pid}"] = k in cr
                p["uncertain"] = []
                p["version"] += 1
                p["fen_error"] = None
            except ValueError as e:
                p["fen_error"] = f"Invalid FEN: {e}"

        with st.container(border=True):
            h = st.columns([0.6, 5, 2.4, 1.2, 0.7])
            h[0].markdown(f"**#{num}**")
            h[1].text_input("Title", key=f"title_{pid}", label_visibility="collapsed")
            h[2].markdown(f"<div class='small-muted' style='padding-top:.45rem'>{p['source']}</div>",
                          unsafe_allow_html=True)
            h[3].checkbox("Export", key=f"inc_{pid}")
            h[4].button("✕", key=f"del_{pid}", help="Remove this board", on_click=remove_puzzle, args=(pid,))

            c0, c1, c2 = st.columns([1.05, 1.5, 1.45], gap="medium")
            with c0:
                st.caption("Original")
                st.image(p["crop"], width="stretch")
                if p["caption"]:
                    st.caption(re.sub(r"^(\d+)\.", r"\1\\.", p["caption"][:200]))  # keep "1." from becoming a list
            with c1:
                bkey = f"board_{pid}_{p['version']}"
                _board(placement=p["placement"], uncertain=p["uncertain"], version=p["version"],
                       flipped=p["flipped"], square=44, key=bkey, default=None)
            with c2:
                st.radio("To move", ["White", "Black"], key=f"stm_{pid}", horizontal=True)
                if p["flipped"]:
                    st.caption("Shown from Black's side in the original, so it was rotated (Black to move assumed).")
                poss = castling_possible(p["placement"])
                st.markdown("<div class='small-muted'>Castling rights</div>", unsafe_allow_html=True)
                cc = st.columns(2)
                for i_, (k, lab) in enumerate(zip("KQkq", ["White O-O", "White O-O-O", "Black O-O", "Black O-O-O"])):
                    cc[i_ % 2].checkbox(lab, key=f"cs{k}_{pid}", disabled=not poss[k])
                fen = current_fen(p)
                ss[fkey] = fen
                p["fen_shown"] = fen
                st.text_input("FEN", key=fkey, help="Edit and press Enter to replace the board")
                st.text_area("Solution (optional)", key=f"sol_{pid}", height=68,
                             placeholder="e.g. 1. Rxh7+ Kxh7 2. Qh5+ Kg8 3. Ng5")
                problems = R.check_position(fen)
                if p["fen_error"]:
                    st.error(p["fen_error"])
                if problems:
                    st.error("⚠ " + " · ".join(problems))
                else:
                    err, san = check_solution(fen, ss[f"sol_{pid}"])
                    if err:
                        st.error("⚠ Solution: " + err)
                    else:
                        msg = "✓ Legal position" + (f" · solution OK: {san}" if san else "")
                        if p["uncertain"]:
                            st.warning(msg + f" · {len(p['uncertain'])} square(s) worth a second look (orange)")
                        else:
                            st.success(msg)
                st.link_button("Open in Lichess ↗", "https://lichess.org/analysis/standard/" + fen.replace(" ", "_"))

# ----------------------------------------------------------------------------- export (rendered last so it sees every edit)
with st.sidebar:
    st.header("Export")
    name = st.text_input("File name", value="puzzles")
    chosen = [p for p in ss.puzzles if ss.get(f"inc_{p['id']}", True)]
    items = [{"fen": current_fen(p), "solution": ss.get(f"sol_{p['id']}", ""), "title": ss.get(f"title_{p['id']}"),
              "source": p["source"]} for p in chosen]
    bad = sum(1 for it in items if R.check_position(it["fen"]))
    pgn = R.make_pgn(items, event=name or "Puzzles") if items else ""
    st.download_button(f"⬇ Download PGN ({len(items)} puzzles)", data=pgn.encode(),
                       file_name=f"{(name or 'puzzles').strip()}.pgn", mime="application/x-chess-pgn",
                       type="primary", disabled=not items, width="stretch")
    if bad:
        st.warning(f"{bad} board(s) are not legal positions yet.")
    if items:
        with st.expander("Show / copy PGN"):
            st.code(pgn, language=None)
    st.divider()
    if ss.puzzles and st.button("Remove all boards", width="stretch"):
        ss.puzzles = []
        ss.pages = []
        ss.upk += 1   # fresh, empty uploader (files already processed stay marked as done)
        st.rerun()
    st.caption("Tip: orange squares on a board are ones the recogniser was unsure about. "
               "Side to move is read from PDF captions such as “Black to move”; otherwise check it.")

"""
Chess Diagram -> PGN  (local web app)

Run:  python app.py      then open http://127.0.0.1:5000
"""
import io
import threading
import uuid

import chess
import chess.pgn
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

import recognize as R

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB uploads

PAGES = {}   # page_id -> BGR image
CROPS = {}   # puzzle_id -> PNG bytes
JOBS = {}    # job_id -> dict(progress, total, puzzles, pages, done, error)
LOCK = threading.Lock()


def _png(img, max_side=None):
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


def _register(puzzles, page_id):
    out = []
    for p in puzzles:
        pid = uuid.uuid4().hex[:10]
        CROPS[pid] = _png(p.pop("crop"), 700)
        p["id"] = pid
        p["page_id"] = page_id
        out.append(p)
    return out


def _decode_image(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # flatten transparency onto white
        a = img[:, :, 3:].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return img


def _run_job(job_id, files):
    job = JOBS[job_id]
    try:
        total = 0
        for name, data in files:
            if name.lower().endswith(".pdf"):
                import pymupdf as fitz
                total += min(len(fitz.open(stream=data, filetype="pdf")), 300)
            else:
                total += 1
        job["total"] = total
        done = 0
        for name, data in files:
            if name.lower().endswith(".pdf"):
                base = done

                def prog(i, n, base=base):
                    job["progress"] = base + i

                import pymupdf as fitz
                doc = fitz.open(stream=data, filetype="pdf")
                for pno in range(min(len(doc), 300)):
                    page_puz, pages = R.process_pdf_page(doc, pno)
                    page_id = uuid.uuid4().hex[:10]
                    PAGES[page_id] = pages
                    job["pages"].append({"id": page_id, "label": f"{name} – page {pno + 1}",
                                         "w": pages.shape[1], "h": pages.shape[0]})
                    for p in page_puz:
                        p["source"] = f"{name}, page {pno + 1}"
                    job["puzzles"] += _register(page_puz, page_id)
                    done += 1
                    job["progress"] = done
            else:
                img = _decode_image(data)
                if img is None:
                    job["warnings"].append(f"Could not read {name}")
                    done += 1
                    continue
                page_id = uuid.uuid4().hex[:10]
                PAGES[page_id] = img
                job["pages"].append({"id": page_id, "label": name, "w": img.shape[1], "h": img.shape[0]})
                puz = R.process_image(img, source=name)
                if not puz:
                    job["warnings"].append(f"No board found in {name} – use “Select a diagram manually”.")
                job["puzzles"] += _register(puz, page_id)
                done += 1
                job["progress"] = done
    except Exception as e:  # pragma: no cover
        job["error"] = f"{type(e).__name__}: {e}"
    job["done"] = True


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/upload")
def upload():
    files = [(f.filename or "pasted.png", f.read()) for f in request.files.getlist("files")]
    if not files:
        return jsonify(error="No files"), 400
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = dict(progress=0, total=len(files), puzzles=[], pages=[], done=False, error=None, warnings=[])
    threading.Thread(target=_run_job, args=(job_id, files), daemon=True).start()
    return jsonify(job=job_id)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    since = int(request.args.get("since", 0))
    psince = int(request.args.get("psince", 0))
    return jsonify(progress=job["progress"], total=job["total"], done=job["done"], error=job["error"],
                   warnings=job["warnings"], puzzles=job["puzzles"][since:], pages=job["pages"][psince:],
                   n_puzzles=len(job["puzzles"]), n_pages=len(job["pages"]))


@app.get("/api/crop/<pid>.png")
def crop(pid):
    return send_file(io.BytesIO(CROPS[pid]), mimetype="image/png")


@app.get("/api/page/<page_id>.png")
def page(page_id):
    return send_file(io.BytesIO(_png(PAGES[page_id], 2000)), mimetype="image/png")


@app.post("/api/manual")
def manual():
    d = request.get_json()
    img = PAGES[d["page_id"]]
    corners = np.float32(d["corners"])  # in original pixel coordinates
    puz = R.process_image(img, source=d.get("source", ""), manual_corners=corners)
    return jsonify(puzzles=_register(puz, d["page_id"]))


@app.post("/api/validate")
def validate():
    d = request.get_json()
    fen = d.get("fen", "")
    problems = R.check_position(fen)
    sol = (d.get("solution") or "").strip()
    sol_err, san = None, ""
    if not problems and sol:
        try:
            pgn = R.make_pgn([{"fen": fen, "solution": sol}])
            if "[unparsed solution" in pgn:
                tok = pgn.split("[unparsed solution from '")[1].split("'")[0]
                sol_err = f"“{tok}” is not a legal move here"
            b = chess.Board(fen)
            g = chess.pgn.read_game(io.StringIO(pgn))
            san = b.variation_san(list(g.mainline_moves())) if g else ""
        except Exception as e:
            sol_err = str(e)
    return jsonify(problems=problems, solution_error=sol_err, solution_san=san)


@app.post("/api/export")
def export():
    d = request.get_json()
    text = R.make_pgn(d["puzzles"], event=d.get("event") or "Puzzles")
    return send_file(io.BytesIO(text.encode()), mimetype="application/x-chess-pgn",
                     as_attachment=True, download_name=(d.get("filename") or "puzzles") + ".pgn")


if __name__ == "__main__":
    import argparse
    import webbrowser

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    R.load_model()
    url = f"http://{a.host}:{a.port}"
    print(f"\n  Chess Diagram -> PGN running at {url}\n")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, debug=False, threaded=True)

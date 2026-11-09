# Chess Diagram → PGN

A small web app that runs on your own computer. It turns chess **puzzle diagrams** into a **PGN file**, from any of these sources:

- PDFs of puzzle books (it reads every page),
- scanned pages and phone photos of books,
- screenshots from chess apps and websites (you can paste them straight in with Ctrl/⌘+V).

It finds every diagram on a page automatically and recognises the pieces. You get a board editor to fix anything it got wrong, then export a single PGN where each puzzle is its own game (`[SetUp "1"]` + `[FEN "..."]`). That PGN imports straight into Lichess Studies, ChessBase, SCID, Chess.com, etc.

No internet connection or GPU is needed, and you don't need to install a deep-learning framework. It only uses OpenCV, NumPy, python-chess, PyMuPDF, and Streamlit or Flask.

It comes in two forms that share the same engine:

- **Streamlit app** (`streamlit_app.py`): put it on the web for free with Streamlit Community Cloud, straight from GitHub.
- **Local app** (`app/app.py`): runs on your own computer with Flask, and supports pasting screenshots with Ctrl/⌘+V.

---

## Put it on the web (free): GitHub + Streamlit Community Cloud

1. **Create a GitHub repository.** On github.com click **New repository** and give it a name, e.g. `chess2pgn`.
   - Make it **Public** for the simplest set-up.
   - Leave "Add a README" unticked.
2. **Upload the files.** On the new repo's page, click **uploading an existing file**. Drag in *everything inside* this folder: `streamlit_app.py`, `requirements.txt`, and the `app`, `components`, `samples`, `train` and `.streamlit` folders. Then click **Commit changes**.
   - `.streamlit` is a hidden folder. If your file browser doesn't show it, it only sets the app's colours and upload limit, so it's optional.
   - With git, the equivalent is:
     ```bash
     git init && git add . && git commit -m "Chess Diagram to PGN"
     git branch -M main
     git remote add origin https://github.com/<you>/chess2pgn.git
     git push -u origin main
     ```
3. **Deploy.** Go to **https://share.streamlit.io**, sign in with GitHub, and click **Create app → Deploy a public app from GitHub**.
   - Repository: `<you>/chess2pgn`, Branch: `main`, Main file path: `streamlit_app.py`.
   - Optionally choose an app URL, e.g. `chess2pgn.streamlit.app`, then click **Deploy**.
4. Wait about 2–3 minutes while it installs. Your app is then live at `https://<name>.streamlit.app`, and you can share that link with anyone.

To update the app later, change a file on GitHub (edit or upload) and the app redeploys automatically.

Good to know about the free tier:
- An app that gets no visitors for 12 hours goes to sleep. The next visitor clicks "wake up" and waits about 30 seconds.
- Memory is limited to roughly 1–2.7 GB, which is plenty for this app.
- Very large books are processed page by page with a progress bar. Up to 400 pages per PDF are scanned.

To run the Streamlit version on your own computer:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Quick start (local Flask app)

You need **Python 3.9 or newer**.

**Windows:** double-click `run.bat`.  
**macOS / Linux:** run `./run.sh` in a terminal.

The first run creates a virtual environment and installs the dependencies. After that, the app opens in your browser at http://127.0.0.1:5000.

To set it up manually instead:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/app.py                  # options: --port 8000  --no-browser
```

## Using it

1. **Upload:** drop PDFs or images onto the page, click *browse*, or paste a screenshot.
2. **Review:** every diagram appears as a card with the original next to the recognised board.
   - Squares the recogniser was unsure about have an **orange outline**.
   - The status line tells you if the position is illegal. Typical messages are "No black king" or "Side not to move is in check", which usually means the side to move is wrong.
   - **Fixing pieces:** choose a piece in the toolbar and click squares to place it. Use *Erase* or right-click to clear a square. With *Move* selected you can drag pieces around, and you can also drag from the toolbar onto the board.
   - **Rotate 180°:** use this if a diagram was printed from Black's side and the tool didn't notice. It usually detects this automatically.
   - **To move / Castling:** set these if needed. For PDFs that contain text, captions like "White to move" or "Black to play" are read automatically.
   - **Solution (optional):** type it in normal notation, e.g. `1. Rxh7+ Kxh7 2. Qh5+ Kg8 3. Ng5`. Each move is checked for legality, and a solution ending in mate sets the result automatically.
3. **Missed a diagram?** Click the page name under *"Missed a diagram?"*, drag a box around the 8×8 squares, and click *Recognise selection*.
4. **Export:** *Download PGN* or *Copy PGN*. Untick *Export* on a card to leave it out. In the Streamlit version these are in the sidebar.

Tip: *Show only boards needing attention* filters the list to cards with orange squares or errors, so a long book is quick to review.

## How it works

```
page image ──► find boards ──► straighten ──► 64 squares ──► classify ──► FEN ──► PGN
   (PDF pages rendered with PyMuPDF)
```

1. **Board detection** (`app/recognize.py`): candidate boxes come from edge maps, thresholds, and "blob" maps. In the blob maps the hatching is filled in, so the dark squares merge into one board shape, which also finds frameless diagrams. Each candidate is then:
   - scored by how well it lines up with an 8×8 grid (edge strength along the 7 internal grid lines plus the light/dark checker pattern),
   - snapped to the grid with a 9-line lattice fit,
   - refined corner by corner.

   This handles perspective in photos as well as page margins and coordinate labels.
2. **Piece recognition:** each square becomes a feature vector (signed-gradient HOG at two scales, plus a 10×10 thumbnail). A small neural network (1686 → 512 → 256 → 13) classifies it, and runs in plain NumPy (`app/model.npz`, 3.7 MB).
3. **Training data:** the network was trained on **20,000 synthetic diagrams** (≈600k squares) generated by `train/synth.py`. The generator mixes:
   - 33 piece sets,
   - book styles (hatched, grey and dotted dark squares; frames; coordinates),
   - app styles (colour themes, highlights, move dots, arrows, coordinates),
   - photo effects (perspective, blur, lighting, noise, JPEG).
4. **Post-processing:** detects boards shown from Black's side, guesses castling rights, checks legality with python-chess, and reads the side to move from PDF captions.

### Measured accuracy (synthetic test sets, random positions)

| Source | Boards found | Squares correct | Boards 100% correct |
|---|---|---|---|
| Book pages (1–6 diagrams per page, 30% photographed) | 94% | 99.9% | 97% |
| App screenshots | 100% | 99.9% | 92% |
| Photos of screens | 96% | 99.1% | 83% |

On a test PDF drawn with a renderer the model never saw, using legal positions from real games, it got 8/8 diagrams exactly right, including side to move. Real positions are easier than the random piece placements used in testing.

## Retraining / adapting to a specific book

If one book's diagram font gives trouble, you can retrain. See `train/README.md`; it takes about 20 minutes on a laptop CPU.

## Limitations

- Side to move cannot be seen in an image. It's read from caption text in PDFs that contain text. Otherwise the default is White, or Black if the diagram is shown from Black's side. Always glance at it.
- Scanned PDFs with no text layer get no caption reading, and the same goes for screenshots (there's no OCR).
- En passant rights are never set. Castling rights are assumed when the king and rook are on their home squares; untick them if needed.
- Very faint or heavily skewed photos may need the manual box.

## Project layout

```
streamlit_app.py  Streamlit version (for Streamlit Community Cloud)
components/       board editor + region selector used by the Streamlit app (plain HTML/JS)
.streamlit/       Streamlit theme and upload limit
app/
  app.py          Flask server (upload, jobs, manual selection, validation, export)
  recognize.py    detection, recognition, FEN/PGN logic
  features.py     square feature extractor (shared with training)
  model.npz       trained network
  static/         web UI (index.html, app.js, style.css, pieces/)
train/
  synth.py, make_dataset.py, train_mlp.py, eval_pipeline.py, rasterize_pieces.py
samples/          a sample PDF, screenshot and photo to try
```

## Credits

The board editor pieces are the "cburnett" set by Colin M.L. Burnett (GPLv2+ / BSD / GFDL / CC BY-SA 3.0).

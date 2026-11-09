# Retraining the piece recogniser

Install the extra training dependency:

```bash
pip install cairosvg
```

### 1. Get piece sets

The training piece images are not included because they come under several licences. Fetch the Lichess piece sets and rasterise them:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/lichess-org/lila.git lila
cd lila && git sparse-checkout set public/piece && cd ..
python rasterize_pieces.py lila/public/piece pieces
```

To add the diagram font from **your own book**, create `pieces/mybook/` containing 12 transparent PNGs named `wK.png wQ.png wR.png wB.png wN.png wP.png bK.png …` (any size; 128×128 works well). Crop them from a scan and erase the background. Also add `"mybook"` to `BOOK_SETS` in `synth.py`, so it is used for the book-style diagrams.

### 2. Generate data and train (run inside `train/`)

```bash
python make_dataset.py 10000 data/train.npz 1000000
python make_dataset.py 10000 data/train2.npz 2000000
python make_dataset.py 1000  data/val.npz 0
python train_mlp.py data/train.npz,data/train2.npz data/val.npz ../app/model.npz 14
```

Training needs about 4 GB of RAM and takes about 15 minutes on 2 CPU cores. It should reach roughly 99.5% validation accuracy per square.

### 3. Check end-to-end accuracy

```bash
python eval_pipeline.py 25
```

`python synth.py` has no CLI. Import it and call `make_sample()` if you want to look at generated examples.

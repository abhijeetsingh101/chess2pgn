"""Rasterize SVG piece sets to 128px RGBA PNGs for synthetic training data."""
import os, sys, cairosvg
SRC = sys.argv[1]; DST = sys.argv[2]
SKIP = {"disguised", "letter", "mono", "shapes", "xkcd", "pixel", "kiwen-suwi"}
names = ["wK","wQ","wR","wB","wN","wP","bK","bQ","bR","bB","bN","bP"]
for s in sorted(os.listdir(SRC)):
    d = os.path.join(SRC, s)
    if s in SKIP or not os.path.isdir(d): continue
    if not all(os.path.exists(os.path.join(d, n + ".svg")) for n in names): continue
    os.makedirs(os.path.join(DST, s), exist_ok=True)
    for n in names:
        cairosvg.svg2png(url=os.path.join(d, n + ".svg"), write_to=os.path.join(DST, s, n + ".png"), output_width=128, output_height=128)
    print(s)
